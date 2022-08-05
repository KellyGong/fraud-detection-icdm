from dataclasses import dataclass
import os
import os.path as osp
import argparse
import json
from tabnanny import verbose

from utils import EarlyStop, setup_seed

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from sklearn.metrics import average_precision_score

from model import RGCN, RGPRGNN, RGAT, Node_Transformation
import nni
import wandb
import random


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dataset/pyg_data/icdm2022_session1.pt')
# parser.add_argument('--dataset', type=str, default='dataset/pyg_data/icdm2022_session1_debug.pt')
parser.add_argument('--labeled-class', type=str, default='item')
parser.add_argument("--batch_size", type=int, default=64,
                    help="Mini-batch size. If -1, use full graph training.")
parser.add_argument("--model", choices=["RGCN", "RGPRGNN", "RGAT"], default="RGPRGNN")
parser.add_argument("--fanout", type=int, default=150,
                    help="Fan-out of neighbor sampling.")
parser.add_argument("--n_layers", type=int, default=3,
                    help="number of propagation rounds")
parser.add_argument("--h_dim", type=int, default=64,
                    help="number of hidden units")
parser.add_argument("--in-dim", type=int, default=256,
                    help="number of hidden units")
parser.add_argument("--n_bases", type=int, default=8,
                    help="number of filter weight matrices, default: -1 [use all]")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--activation", choices=['relu', 'leaklyrelu', 'elu'], default='relu')
parser.add_argument("--label_smoothing", type=float, default=0)

parser.add_argument("--validation", type=bool, default=True)
parser.add_argument("--early_stopping", type=int, default=5)
parser.add_argument("--n-epoch", type=int, default=15)
parser.add_argument("--test-file", type=str, default="dataset/icdm2022_session1_test_ids.txt")
parser.add_argument("--json-file", type=str, default="pyg_pred_session1.json")
parser.add_argument("--inference", type=bool, default=False)
# parser.add_argument("--record-file", type=str, default="record.txt")

# pseudo label training
parser.add_argument("--pseudo_positive", type=int, default=500)
parser.add_argument("--pseudo_negative", type=int, default=2000)
parser.add_argument("--pseudo", action='store_true', default=False)

parser.add_argument("--pre_transform", action='store_true', default=False)
parser.add_argument("--alpha", type=float, default=0.1)

parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--device", type=str, default="cuda:1")

# grid search hyperparameters
parser.add_argument("--nni", action='store_true', default=False)
parser.add_argument("--wandb", action='store_true', default=False)
parser.add_argument("--debug", action='store_true', default=False)

args = parser.parse_args()

if args.nni:
    params = nni.get_next_parameter()
    old_params = vars(args)
    old_params.update(params)
    args = argparse.Namespace(**old_params)

if args.wandb:
    wandb.init(project='icdm2022',
               tags=[f'{args.model}'],
               entity='gztql',
               config=vars(args))
    args = argparse.Namespace(**wandb.config)

model_id = random.randint(0, 100000)
model_path = osp.join('best_model', args.model + "_" + str(model_id) + ".pth")

print(model_path)

setup_seed(2022)

device = torch.device(f'{args.device}' if torch.cuda.is_available() else 'cpu')
hgraph = torch.load(args.dataset)

labeled_class = args.labeled_class

# print(hgraph.node_types)

if args.inference == False:
    # train_idx = hgraph[labeled_class].pop('train_idx')
    train_idx = hgraph[labeled_class]['train_idx']
    if args.validation:
        # val_idx = hgraph[labeled_class].pop('val_idx')
        val_idx = hgraph[labeled_class]['val_idx']

train_idx = torch.cat((train_idx, val_idx))

test_id = [int(x) for x in open(args.test_file).readlines()]
converted_test_id = []
for i in test_id:
    converted_test_id.append(hgraph['item'].maps[i])
 
test_idx = torch.LongTensor(converted_test_id)


nolabel_idx = np.array([i for i in range(hgraph[labeled_class]['y'].shape[0])])
nolabel_idx = np.setdiff1d(nolabel_idx, train_idx.numpy(), True)
nolabel_idx = np.setdiff1d(nolabel_idx, val_idx.numpy(), True)
nolabel_idx = np.setdiff1d(nolabel_idx, test_idx.numpy(), True)
nolabel_idx = torch.LongTensor(nolabel_idx)

# C class balance parameter

C = len(np.where(hgraph[labeled_class]['y'].numpy() == 1)[0]) / len(np.where(hgraph[labeled_class]['y'].numpy() == 0)[0])
class_balance_ratio = torch.tensor([1, C], device=device, requires_grad=False)

args.pseudo_negative = int(args.pseudo_positive / C)

num_node_types = len(hgraph.node_types)

print(args)

def gen_dataloader(hgraph, labeled_class, idx, args, shuffle=False, balance=False):
    if balance:
        samples = idx.numpy()

        positive_indices = samples[np.in1d(samples, np.where(hgraph[labeled_class]['y'].numpy() == 1)[0])] 
        negative_indices = samples[np.in1d(samples, np.where(hgraph[labeled_class]['y'].numpy() == 0)[0])]  

        negative_indices = np.random.choice(negative_indices, len(positive_indices))
        new_idx = np.concatenate([positive_indices, negative_indices])
        np.random.shuffle(new_idx)
        new_idx = torch.tensor(new_idx)

        # print(torch.count_nonzero(hgraph[labeled_class]['x'][idx]).item())

        dataloader = NeighborLoader(hgraph,
                                    input_nodes=(labeled_class, new_idx),
                                    num_neighbors=[args.fanout] * args.n_layers,
                                    shuffle=shuffle,
                                    batch_size=args.batch_size)
    
    else:
        dataloader = NeighborLoader(hgraph,
                                    input_nodes=(labeled_class, idx),
                                    num_neighbors=[args.fanout] * args.n_layers,
                                    shuffle=shuffle,
                                    batch_size=args.batch_size, num_workers=4)
    return dataloader


# # No need to maintain these features during evaluation:
# # Add global node index information.
# test_loader.data.num_nodes = data.num_nodes
# test_loader.data.n_id = torch.arange(data.num_nodes)


num_relations = len(hgraph.edge_types)

if args.inference:
    model = torch.load(osp.join('best_model', args.model + ".pth"), map_location=args.device)
else:
    if args.model == 'RGCN':
        model = RGCN(in_channels=args.in_dim,
                     hidden_channels=args.h_dim,
                     out_channels=2,
                     num_relations=num_relations,
                     num_bases=args.n_bases,
                     n_layers=args.n_layers,
                     dropout=args.dropout)
    
    elif args.model == 'RGAT':
        model = RGAT(in_channels=args.in_dim,
                     hidden_channels=args.h_dim,
                     out_channels=2,
                     num_relations=num_relations,
                     num_bases=args.n_bases,
                     n_layers=args.n_layers)
    
    elif args.model == 'RGPRGNN':
        model = RGPRGNN(in_channels=args.h_dim if args.pre_transform else args.in_dim,
                        hidden_channels=args.h_dim,
                        out_channels=2,
                        num_relations=num_relations,
                        num_bases=args.n_bases,
                        alpha=args.alpha,
                        n_layers=args.n_layers,
                        dropout=args.dropout,
                        pre_transform=args.pre_transform)
    
    else:
        raise NotImplementedError

    # node_transformation = Node_Transformation(args.in_dim, args.h_dim, num_node_types)

    model = model.to(device)
    if args.pre_transform:
        node_transformation = Node_Transformation(args.in_dim, args.h_dim, num_node_types).to(device)
        params = list(model.parameters()) + list(node_transformation.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()

    train_loader = gen_dataloader(hgraph, labeled_class, train_idx, args, shuffle=True, balance=False)

    pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in train_loader:
        optimizer.zero_grad()
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)

        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()

        if args.pre_transform:
            item_id = batch._node_type_names.index(args.labeled_class)
            batch.x = node_transformation(batch.x.to(device), batch.node_type.to(device), item_id)

        y_hat = model(batch.x.to(device), 
                      batch.edge_index.to(device),
                      batch.edge_type.to(device))[start:start + batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        pbar.update(batch_size)
        if args.debug:
            break
    pbar.close()
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())

    return total_loss / total_examples, total_correct / total_examples, ap_score


@torch.no_grad()
def val():
    model.eval()
    val_loader = gen_dataloader(hgraph, labeled_class, val_idx, args, shuffle=False)
    pbar = tqdm(total=int(len(val_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in val_loader:
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()

        # if args.pre_transform:
        #     batch.x = node_transformation(batch.x.to(device), batch.node_type.to(device))
        if args.pre_transform:
            item_id = batch._node_type_names.index(args.labeled_class)
            batch.x = node_transformation(batch.x.to(device), batch.node_type.to(device), item_id)

        y_hat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))[
                start:start + batch_size]
        loss = F.cross_entropy(y_hat, y)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        pbar.update(batch_size)
        if args.debug:
            break
    pbar.close()
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())

    return total_loss / total_examples, total_correct / total_examples, ap_score


@torch.no_grad()
def pseudo_label_gen():
    global train_idx, nolabel_idx

    model.eval()
    test_loader = gen_dataloader(hgraph, labeled_class, nolabel_idx, args, shuffle=True)
    pbar = tqdm(total=int(len(test_loader.dataset)), ascii=True)
    pbar.set_description(f'Generate Pseudo Label')
    y_pred = []
    for i, batch in enumerate(test_loader):
        batch_size = batch[labeled_class].batch_size
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()

        if args.pre_transform:
            batch.x = node_transformation(batch.x.to(device), batch.node_type.to(device))

        y_hat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))[
                start:start + batch_size]
        pbar.update(batch_size)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        if args.debug or i > 2000:
            break
    pbar.close()

    y_pred_tensor = torch.cat(y_pred)
    top_abnormal_pred, top_abnormal_indices = torch.sort(y_pred_tensor, descending=True)
    top_abnormal_idx = batch.n_id[top_abnormal_indices]

    top_normal_pred, top_normal_indices = torch.sort(y_pred_tensor, descending=False)
    top_normal_idx = batch.n_id[top_normal_indices]

    iteration_top_abnormal_idx = top_abnormal_idx[:args.pseudo_positive]
    iteration_top_normal_idx = top_normal_idx[:args.pseudo_negative]

    # add new pseudo labels to graph for new training
    hgraph[labeled_class]['y'][iteration_top_abnormal_idx] = 1
    hgraph[labeled_class]['y'][iteration_top_normal_idx] = 0

    train_idx = torch.cat([train_idx, iteration_top_abnormal_idx, iteration_top_normal_idx])

    nolabel_idx = np.setdiff1d(nolabel_idx.numpy(), iteration_top_abnormal_idx.numpy(), True)
    nolabel_idx = np.setdiff1d(nolabel_idx, iteration_top_normal_idx.numpy(), True)
    nolabel_idx = torch.LongTensor(nolabel_idx)


@torch.no_grad()
def test():
    model.eval()
    test_loader = gen_dataloader(hgraph, labeled_class, test_idx, args, shuffle=False)
    pbar = tqdm(total=int(len(test_loader.dataset)), ascii=True)
    pbar.set_description(f'Generate Final Result:')
    y_pred = []
    for batch in test_loader:
        batch_size = batch[labeled_class].batch_size
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()
        if args.pre_transform:
            item_id = batch._node_type_names.index(args.labeled_class)
            batch.x = node_transformation(batch.x.to(device), batch.node_type.to(device), item_id)
        y_hat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))[
                start:start + batch_size]
        pbar.update(batch_size)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
    pbar.close()

    return torch.hstack(y_pred)


if args.inference == False:
    print("Start training")
    best_val_ap = 0
    earlystop = EarlyStop(interval=args.early_stopping)
    for epoch in range(1, args.n_epoch + 1):
        train_loss, train_acc, train_ap = train(epoch)
        print(f'Train: Epoch {epoch:02d}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AP_Score: {train_ap:.4f}')

        if args.validation:
            val_loss, val_acc, val_ap = val()
            print(f'Val: Epoch: {epoch:02d}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AP_Score: {val_ap:.4f}')

            # nni
            if args.nni:
                nni.report_intermediate_result({
                    "default": val_ap,
                    "loss": val_loss,
                    "acc": val_acc
                })
            
            # wandb
            if args.wandb:
                wandb.log({
                    "val_ap": val_ap,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)

            # save model
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                torch.save(model, model_path)
            
            # early stop
            if earlystop.update(val_ap):
                print("Early Stopping")
                break
        
        # add pseudo label to the training set
        if args.pseudo and epoch % 3 == 0:
            pseudo_label_gen()
        

    print(f"Complete Training (best val_ap: {best_val_ap})")

    if args.nni:
        nni.report_final_result(best_val_ap)

    if args.wandb:
        wandb.run.summary['best_val_ap'] = best_val_ap


#    with open(args.record_file, 'a+') as f:
#        f.write(f"{args.model_id:2d} {args.h_dim:3d} {args.n_layers:2d} {args.lr:.4f} {end:02d} {float(val_ap_list[-1]):.4f} {np.argmax(val_ap_list)+5:02d} {float(np.max(val_ap_list)):.4f}\n")


# if args.inference == True:

model = torch.load(model_path, map_location=args.device)
print(model_path)
y_pred = test()
result_path = osp.join('best_result', "pyg_pred_session1_" + args.model + "_" + str(model_id) + ".json")
with open(result_path, 'w+') as f:
    for i in range(len(test_id)):
        y_dict = {}
        y_dict["item_id"] = int(test_id[i])
        y_dict["score"] = float(y_pred[i])
        json.dump(y_dict, f)
        f.write('\n')
