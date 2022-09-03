from dataclasses import dataclass
import os
import os.path as osp
import argparse
import json
from tabnanny import verbose
import time
import datetime

from utils import EarlyStop, setup_seed

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import permutations
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborLoader, HGTLoader
import torch_geometric.transforms as T
from sklearn.metrics import average_precision_score, confusion_matrix

from torch.nn import Linear
from torch_geometric.nn import RGCNConv

from model import RGCN, RGPRGNN, RGAT, Node_Transformation, HGT, ResRGCN, Post_Transformation, RFILM
import nni
import wandb
import random
from info_nce import InfoNCE
from losses import SupConLoss, focal_loss


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dataset/pyg_data/icdm2022_session1.pt')
parser.add_argument('--labeled-class', type=str, default='item')
parser.add_argument("--batch_size", type=int, default=256,
                    help="Mini-batch size. If -1, use full graph training.")
parser.add_argument("--model", choices=["RGCN", "RGPRGNN", "RGAT", "HGT", "ResRGCN", "RFILM"], default="RGPRGNN")
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
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--activation", choices=['relu', 'leaklyrelu', 'elu'], default='relu')
parser.add_argument("--label_smoothing", type=float, default=0)

parser.add_argument("--validation", type=bool, default=True)
parser.add_argument("--early_stopping", type=int, default=10)
parser.add_argument("--n-epoch", type=int, default=100)
parser.add_argument("--test-file", type=str, default=None)
parser.add_argument("--json-file", type=str, default="pyg_pred_session1.json")
parser.add_argument("--inference", type=bool, default=False)
# parser.add_argument("--record-file", type=str, default="record.txt")

parser.add_argument("--dropedge", type=float, default=0.2)
parser.add_argument("--drop_distance", action='store_true', default=False)

# sample unbalance hyperparameter
parser.add_argument("--balance", type=bool, default=False)
parser.add_argument("--focal", type=bool, default=False)
parser.add_argument("--positive_weight", type=float, default=0.8)
parser.add_argument("--val_positive_rate", type=float, default=0.0625)

# pseudo label training
parser.add_argument("--pseudo_positive", type=int, default=500)
parser.add_argument("--pseudo_negative", type=int, default=2000)
parser.add_argument("--pseudo", action='store_true', default=False)

# contrastive learning
parser.add_argument("--cl", action='store_true', default=True)
parser.add_argument("--cl_supervised", action='store_true', default=False)
parser.add_argument("--cl_joint_loss", action='store_true', default=True)
parser.add_argument("--cl_epoch", type=int, default=3)
parser.add_argument("--cl_lr", type=float, default=0.002)
parser.add_argument("--cl_finetune_lr", type=float, default=0.005)
parser.add_argument("--cl_common_lr", type=float, default=0.002)
parser.add_argument("--cl_batch", type=int, default=2048)

# build item-item relation through feature proximity and metapath (common neighbor b)
parser.add_argument("--item_item", action='store_true', default=False)
parser.add_argument("--node_sample", type=int, default=80000)
parser.add_argument("--edge_add", type=int, default=500000)
parser.add_argument("--metapath", type=bool, default=False)
parser.add_argument("--meta_fraction", type=float, default=0.1)

parser.add_argument("--pre_transform", action='store_true', default=False)
parser.add_argument("--alpha", type=float, default=0.5)

parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--model_id", type=int, default=1)

# grid search hyperparameters
parser.add_argument("--nni", action='store_true', default=False)
parser.add_argument("--wandb", action='store_true', default=True)
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
model_path = args.model + "_" + str(model_id) + ".pth"

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

test_id = [int(x) for x in open(args.test_file).readlines()]
converted_test_id = []
for i in test_id:
    converted_test_id.append(hgraph['item'].maps[i])
 
test_idx = torch.LongTensor(converted_test_id)

all_id = torch.cat([train_idx, val_idx, test_idx])


nolabel_idx = np.array(range(hgraph[labeled_class]['y'].shape[0]))
nolabel_idx = np.setdiff1d(nolabel_idx, train_idx.numpy(), True)
nolabel_idx = np.setdiff1d(nolabel_idx, val_idx.numpy(), True)
nolabel_idx = np.setdiff1d(nolabel_idx, test_idx.numpy(), True)
nolabel_idx = torch.LongTensor(nolabel_idx)

# C class balance parameter

C = len(np.where(hgraph[labeled_class]['y'].numpy() == 1)[0]) / len(np.where(hgraph[labeled_class]['y'].numpy() == 0)[0])
class_balance_ratio = torch.tensor([1 - args.positive_weight, args.positive_weight], device=device, requires_grad=False)


def refine_positive_rate(positive_ids, negative_ids, val_positive_rate=args.val_positive_rate, val_rate=0.2):
    all_id_len = len(positive_ids) + len(negative_ids)
    np.random.shuffle(positive_ids)
    np.random.shuffle(negative_ids)
    val_positive_len = int(all_id_len * val_rate * val_positive_rate)
    val_negative_len = int(all_id_len * val_rate * (1 - val_positive_rate))
    val_idx = torch.tensor(np.concatenate((positive_ids[:val_positive_len], negative_ids[:val_negative_len])))
    train_idx = torch.tensor(np.concatenate((positive_ids[val_positive_len:], negative_ids[val_negative_len:])))
    return train_idx, val_idx

if args.balance:
    train_idx, val_idx = refine_positive_rate(np.where(hgraph[labeled_class]['y'].numpy() == 1)[0], np.where(hgraph[labeled_class]['y'].numpy() == 0)[0])

# args.pseudo_negative = int(args.pseudo_positive / C)

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
        # dataloader = HGTLoader(hgraph,
        #                        input_nodes=(labeled_class, idx),
        #                        num_samples=[500] * 5,
        #                        shuffle=shuffle,
        #                        batch_size=args.batch_size, num_workers=4)
    return dataloader


# # No need to maintain these features during evaluation:
# # Add global node index information.
# test_loader.data.num_nodes = data.num_nodes
# test_loader.data.n_id = torch.arange(data.num_nodes)

###################### add a new edge_type item-I-item based on feature distance
def add_item_item_edge(hgraph, num_node2sample, num_edge2add, train_idx, val_idx, test_idx):
    start_time = time.time()
    train_size, val_size, test_size, all_size = torch.numel(train_idx), torch.numel(val_idx), torch.numel(test_idx), torch.numel(train_idx) + torch.numel(val_idx) + torch.numel(test_idx)
    # train_p, val_p, test_p = 0.56, 0.14, 0.3
    train_p, val_p, test_p = train_size / all_size, val_size / all_size, test_size / all_size
    print('Adding {} edges item-I-item among {} nodes...'.format(num_edge2add, num_node2sample))
    train_sample_idx = train_idx[np.random.choice(len(train_idx), int(train_p*num_node2sample), replace=False)]
    val_sample_idx = val_idx[np.random.choice(len(val_idx), int(val_p*num_node2sample), replace=False)]
    test_sample_idx = test_idx[np.random.choice(len(test_idx), int(test_p*num_node2sample), replace=False)]
    all_sample_idx = torch.cat([train_sample_idx, val_sample_idx, test_sample_idx])
    # distance
    dist = torch.cdist(hgraph['item'].x[all_sample_idx], hgraph['item'].x[all_sample_idx])
    dist[torch.triu(dist, diagonal=1)==0] = -1 # remove lower triangle
    dist[dist==0] = -1 # remove nodes that have all 0 feature

    # select top similar node pair 
    dist = np.array(dist)
    off_diag_ind = dist!=-1
    off_diag = dist[off_diag_ind]
    threshold = np.partition(off_diag, num_edge2add // 2)[:num_edge2add//2].max()
    edge2add_idx = np.where((dist<=threshold) & off_diag_ind)

    # add new relation "item-item" to the selected node pairs
    sources, targets = all_sample_idx[edge2add_idx[0]], all_sample_idx[edge2add_idx[1]]
    hgraph['item', 'I', 'item'].edge_index = torch.vstack([torch.cat([sources, targets]), torch.cat([targets, sources])])
    print("It took {} minutes to add {} edges among {} nodes".format((time.time() - start_time)/60, hgraph['item', 'I', 'item'].edge_index.shape[1], torch.numel(all_sample_idx)))
    return hgraph

if args.item_item:
    hgraph = add_item_item_edge(hgraph, args.node_sample, args.edge_add, train_idx, val_idx, test_idx)

###################### add a new edge_type item-I2-item based on common neighbor b
def add_metapath(batch):
    edge_index = to_undirected(batch.edge_index)
    b_mask = batch.node_type == 0
    b_idx = torch.where(b_mask)[0]
    # item_mask = batch.node_type == 3
    # item_idx = torch.where(item_mask)[0]
    # f_mask = batch.node_type == 1
    # f_idx = torch.where(f_mask)[0]
    new_edge_idx = []
    edge_b = (edge_index[0].unsqueeze(1) == b_idx).nonzero() # edge from b
    b_edge_index = edge_index[:, edge_b[:, 0]]
    b_idx, counts = torch.unique(edge_b[:, 1], return_counts=True)
    b_idx, counts = torch.flip(b_idx, [0]), torch.flip(counts, [0]) # b nodes with fewer edges have higher priority
    num_edge = 0
    for b in b_idx[counts>1]: # for each b, add complete subgraph for all its item neighbors
        new_edges = torch.tensor(np.array(list(permutations(b_edge_index[1, edge_b[:, 1] == b].tolist(), 2)))).T
        new_edge_idx.append(new_edges)
        num_edge += new_edges.shape[1]
        if num_edge >= args.meta_fraction * edge_index.shape[1]:
            break
    new_edge_index = torch.cat(new_edge_idx, dim=1)
    batch.edge_index = torch.cat([batch.edge_index, new_edge_index], dim=1)
    batch.edge_type = torch.cat([batch.edge_type, (torch.ones(new_edge_index.shape[1]) * (num_relations-1)).long()], dim=0)
    return batch

num_relations = len(hgraph.edge_types)
if args.metapath:
    num_relations += 1
print(f'num_relation: {num_relations}')
# print(hgraph.metadata())

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
    
    elif args.model == 'RFILM':
        model = RFILM(in_channels=args.h_dim if args.pre_transform else args.in_dim,
                        hidden_channels=args.h_dim,
                        out_channels=2,
                        num_relations=num_relations,
                        num_bases=args.n_bases,
                        alpha=args.alpha,
                        n_layers=args.n_layers,
                        dropout=args.dropout,
                        pre_transform=args.pre_transform)
    
    elif args.model == 'HGT':
        model = HGT(metadata=hgraph.metadata(),
                    in_channels=args.h_dim if args.pre_transform else args.in_dim,
                    hidden_channels=args.h_dim,
                    out_channels=2,
                    num_relations=num_relations,
                    num_bases=args.n_bases,
                    n_layers=args.n_layers,
                    dropout=args.dropout,
                    pre_transform=args.pre_transform)
    
    elif args.model == 'ResRGCN':
        model = ResRGCN(in_channels=args.in_dim,
                        hidden_channels=args.h_dim,
                        out_channels=2,
                        num_relations=num_relations,
                        num_bases=args.n_bases,
                        n_layers=args.n_layers,
                        dropout=args.dropout)
    
    
    
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
    
    if args.cl:
        post_transformation = Post_Transformation(args.h_dim, args.h_dim).to(device)
        model_common_params = list(set(model.parameters()) - set(model.lin2.parameters()))
        

        cl_params = model_common_params + list(post_transformation.parameters())
        cl_optimizer = torch.optim.Adam(cl_params, lr=args.cl_lr)

        optimizer = torch.optim.Adam([{"params": list(model.lin2.parameters()), "lr": args.cl_finetune_lr}, 
                                      {"params": model_common_params, "lr": args.cl_common_lr}])


def augment(batch, augment_type='dropedge', drop_rate=0.2, similarity=True):
    if augment_type == 'dropedge':
        if not similarity:
            num_edge = batch.edge_index.shape[1]
            # keep the edges with labeled items
            # edge_index_in_labeled_item = torch.isin(batch.edge_index, all_id)
            # edge_index_in_labeled_item = (edge_index_in_labeled_item[0] | edge_index_in_labeled_item[1]).int().numpy()
            # edge_indexes = np.where(edge_index_in_labeled_item==0)[0]
            edge_indexes = list(range(num_edge))
            select_edge_indexes = np.random.permutation(edge_indexes)[:int(num_edge * (1 - drop_rate))]
            batch.edge_index = torch.index_select(batch.edge_index, 1, torch.tensor(select_edge_indexes))
            batch.edge_type = torch.index_select(batch.edge_type, 0, torch.tensor(select_edge_indexes))
        
        else:
            num_edge = batch.edge_index.shape[1]
            x_1 = batch.x[batch.edge_index[0]]
            x_2 = batch.x[batch.edge_index[1]]
            edge_dis = torch.norm(x_1 - x_2, p=2, dim=1).numpy()
            select_edge_indexes = np.argpartition(edge_dis, int(num_edge * (1 - drop_rate)))[:int(num_edge * (1 - drop_rate))]
            batch.edge_index = torch.index_select(batch.edge_index, 1, torch.tensor(select_edge_indexes))
            batch.edge_type = torch.index_select(batch.edge_type, 0, torch.tensor(select_edge_indexes))
        return batch


def contrastive():
    for epoch in range(1, args.cl_epoch + 1):
        contrastive_training(epoch)
        contrastive_testing(epoch)


@torch.no_grad()
def contrastive_testing(epoch):
    model.eval()
    post_transformation.eval()
    cl_loss = InfoNCE()
    valid_loader = gen_dataloader(hgraph, labeled_class, val_idx, args, shuffle=False, balance=False)
    pbar = tqdm(total=int(len(valid_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')
    total_examples=total_loss=0
    for batch in valid_loader:
        cl_batch_size = min(args.cl_batch, batch[labeled_class].y.shape[0])
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()

        batch_aug = augment(batch, similarity=args.drop_distance)
        batch = augment(batch, similarity=args.drop_distance)
        y_pretrain = model(batch.x.to(device), 
                           batch.edge_index.to(device),
                           batch.edge_type.to(device), cl=True)[start: start+cl_batch_size]
        y_pretrain = post_transformation(y_pretrain)

        y_pretrain_aug = model(batch_aug.x.to(device), 
                               batch_aug.edge_index.to(device),
                               batch_aug.edge_type.to(device), cl=True)[start: start+cl_batch_size]
        y_pretrain_aug = post_transformation(y_pretrain_aug)

        loss = cl_loss(y_pretrain, y_pretrain_aug)
        pbar.update(batch_size)
        total_examples += cl_batch_size
        total_loss += float(loss) * cl_batch_size

    print(f'epoch: {epoch:02d}, cl test loss: {total_loss / total_examples:.4f}')


def contrastive_training(epoch):
    model.train()
    post_transformation.train()
    unsupervised_criterion = InfoNCE()
    supervised_criterion = SupConLoss(temperature=0.1)
    sample_no_label_id = torch.tensor(np.random.permutation(nolabel_idx.numpy())[:int(train_idx.shape[0] * 3)])
    # train_loader = gen_dataloader(hgraph, labeled_class, torch.cat([train_idx, test_idx]), args, shuffle=True, balance=False)
    train_loader = gen_dataloader(hgraph, labeled_class, train_idx, args, shuffle=True, balance=False)
    pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')
    total_examples=total_loss=0
    for batch in train_loader:
        cl_optimizer.zero_grad()
        batch_size = batch[labeled_class].batch_size
        cl_batch_size = min(args.cl_batch, batch[labeled_class].y.shape[0] - batch_size)
        y = batch[labeled_class].y[:batch_size].to(device)
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()

        # subgraph augmentation
        batch_aug = augment(batch, similarity=args.drop_distance)
        y_pretrain = model(batch.x.to(device), 
                           batch.edge_index.to(device),
                           batch.edge_type.to(device), cl=True)[start: start+cl_batch_size]
        y_pretrain = post_transformation(y_pretrain)

        y_pretrain_aug = model(batch_aug.x.to(device), 
                               batch_aug.edge_index.to(device),
                               batch_aug.edge_type.to(device), cl=True)[start: start+cl_batch_size]
        y_pretrain_aug = post_transformation(y_pretrain_aug)
        
        # supervised InfoNCE for items with labels
        supervised_loss = supervised_criterion(torch.cat([y_pretrain[:batch_size].unsqueeze(1), 
                                                          y_pretrain_aug[:batch_size].unsqueeze(1)], dim=1), y)
        # unsupervised InfoNCE for items without labels
        unsupervised_loss = unsupervised_criterion(y_pretrain[batch_size: batch_size+cl_batch_size], 
                                                   y_pretrain_aug[batch_size: batch_size+cl_batch_size])
        
        if args.cl_joint_loss:
            loss = supervised_loss + unsupervised_loss
        elif args.cl_supervised:
            loss = supervised_loss
        else:
            loss = unsupervised_loss
        
        loss.backward()
        cl_optimizer.step()
        pbar.update(batch_size)
        total_examples += cl_batch_size
        total_loss += float(loss) * cl_batch_size

    print(f'epoch: {epoch:02d}, cl train loss: {total_loss / total_examples:.4f}')


def train(epoch):
    model.train()

    train_loader = gen_dataloader(hgraph, labeled_class, train_idx, args, shuffle=True, balance=False)

    pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')
    loss_fn = focal_loss(num_classes=2)
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
        if args.metapath:
            batch = add_metapath(batch)
        if args.pre_transform:
            item_id = batch._node_type_names.index(args.labeled_class)
            batch.x = node_transformation(batch.x.to(device), batch.node_type.to(device), item_id)

        batch = augment(batch, similarity=args.drop_distance)

        y_hat = model(batch.x.to(device), 
                      batch.edge_index.to(device),
                      batch.edge_type.to(device))[start:start + batch_size]

        # batch.x_dict = {
        #     node_type: x.to(device)
        #     for node_type, x in batch.x_dict.items()
        # }
        # batch.edge_index_dict = {
        #     edge_type: edge_index.to(device)
        #     for edge_type, edge_index in batch.edge_index_dict.items()
        # }

        # y_hat = model(batch.x_dict, batch.edge_index_dict)[:batch_size]

        if args.balance:
            loss = F.cross_entropy(y_hat, y, weight=class_balance_ratio)
        if args.focal:
            loss = loss_fn(y_hat, y)
        else:
            loss = F.cross_entropy(y_hat, y)
        
        loss = loss_fn(y_hat, y)
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
    y_pred_binary = []
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
        if args.metapath:
            batch = add_metapath(batch)
        # if args.pre_transform:
        #     batch.x = node_transformation(batch.x.to(device), batch.node_type.to(device))
        if args.pre_transform:
            item_id = batch._node_type_names.index(args.labeled_class)
            batch.x = node_transformation(batch.x.to(device), batch.node_type.to(device), item_id)

        y_hat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))[
                start:start + batch_size]

        # batch.x_dict = {
        #     node_type: x.to(device)
        #     for node_type, x in batch.x_dict.items()
        # }
        # batch.edge_index_dict = {
        #     edge_type: edge_index.to(device)
        #     for edge_type, edge_index in batch.edge_index_dict.items()
        # }

        # y_hat = model(batch.x_dict, batch.edge_index_dict)[:batch_size]

        loss = F.cross_entropy(y_hat, y)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        y_pred_binary.append(y_hat.argmax(dim=-1).detach().cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        pbar.update(batch_size)
        if args.debug:
            break
    pbar.close()
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())
    confuse_matrix = confusion_matrix(torch.hstack(y_true).numpy(), torch.hstack(y_pred_binary).numpy())

    return total_loss / total_examples, total_correct / total_examples, ap_score, confuse_matrix


@torch.no_grad()
def pseudo_label_gen():
    global train_idx, nolabel_idx

    model.eval()
    test_loader = gen_dataloader(hgraph, labeled_class, torch.cat([train_idx, val_idx]), args, shuffle=True)
    pbar = tqdm(total=int(len(test_loader.dataset)), ascii=True)
    pbar.set_description(f'Generate Pseudo Label')
    y_pred = []
    node_ids = []
    for i, batch in enumerate(test_loader):
        batch_size = batch[labeled_class].batch_size
        node_size = batch[labeled_class].num_nodes
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()

        if args.pre_transform:
            batch.x = node_transformation(batch.x.to(device), batch.node_type.to(device))

        y_hat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))[
                start + batch_size: start + node_size]
        pbar.update(batch_size)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        node_ids.append(batch.n_id[start + batch_size: start + node_size])
        # if args.debug or i > 2000:
        #     break
    pbar.close()

    y_pred_tensor = torch.cat(y_pred)
    item_id_tensor = torch.cat(node_ids)
    _, top_abnormal_indices = torch.sort(y_pred_tensor, descending=True)
    top_abnormal_idx = item_id_tensor[top_abnormal_indices]

    _, top_normal_indices = torch.sort(y_pred_tensor, descending=False)
    top_normal_idx = item_id_tensor[top_normal_indices]

    iteration_top_abnormal_idx = torch.unique(top_abnormal_idx[:args.pseudo_positive])
    iteration_top_normal_idx = torch.unique(top_normal_idx[:args.pseudo_negative])

    # add new pseudo labels to graph for new training
    hgraph[labeled_class]['y'][iteration_top_abnormal_idx] = 1
    hgraph[labeled_class]['y'][iteration_top_normal_idx] = 0

    train_idx = torch.cat([train_idx, iteration_top_abnormal_idx, iteration_top_normal_idx])

    nolabel_idx = np.setdiff1d(nolabel_idx.numpy(), iteration_top_abnormal_idx.numpy(), True)
    nolabel_idx = np.setdiff1d(nolabel_idx, iteration_top_normal_idx.numpy(), True)
    nolabel_idx = torch.LongTensor(nolabel_idx)

    def weight_reset(m):
        if isinstance(m, RGCNConv) or isinstance(m, Linear):
            m.reset_parameters()

    # model reset parameters
    model.apply(weight_reset)


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
        if args.metapath:
            batch = add_metapath(batch)
        if args.pre_transform:
            item_id = batch._node_type_names.index(args.labeled_class)
            batch.x = node_transformation(batch.x.to(device), batch.node_type.to(device), item_id)
        y_hat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))[
                start:start + batch_size]
        # batch.x_dict = {
        #     node_type: x.to(device)
        #     for node_type, x in batch.x_dict.items()
        # }
        # batch.edge_index_dict = {
        #     edge_type: edge_index.to(device)
        #     for edge_type, edge_index in batch.edge_index_dict.items()
        # }

        # y_hat = model(batch.x_dict, batch.edge_index_dict)[:batch_size]

        pbar.update(batch_size)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
    pbar.close()

    return torch.hstack(y_pred)


if args.inference == False:

    if args.cl:
        print("Start contrastive training")
        contrastive()
    
    print("Start training")
    best_val_ap = 0
    best_confuse_matrix = None
    earlystop = EarlyStop(interval=args.early_stopping)
    for epoch in range(1, args.n_epoch + 1):
        train_loss, train_acc, train_ap = train(epoch)
        print(f'Train: Epoch {epoch:02d}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AP_Score: {train_ap:.4f}')

        if args.validation:
            val_loss, val_acc, val_ap, confuse_matrix = val()
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
                best_confuse_matrix = confuse_matrix
                torch.save(model, model_path)
            
            # early stop
            if earlystop.update(val_ap):
                print("Early Stopping")
                break
        
        # add pseudo label to the training set
        if args.pseudo and epoch % 15 == 0:
            pseudo_label_gen()
            earlystop = EarlyStop(interval=args.early_stopping)

    if args.nni:
        nni.report_final_result(best_val_ap)

    if args.wandb:
        wandb.run.summary['best_val_ap'] = best_val_ap

    print(f"Complete Training (best val_ap: {best_val_ap})")
    

#    with open(args.record_file, 'a+') as f:
#        f.write(f"{args.model_id:2d} {args.h_dim:3d} {args.n_layers:2d} {args.lr:.4f} {end:02d} {float(val_ap_list[-1]):.4f} {np.argmax(val_ap_list)+5:02d} {float(np.max(val_ap_list)):.4f}\n")


if args.inference == True:
    model = torch.load(model_path, map_location=args.device)
    print(model_path)
    y_pred = test()
    # result_path = osp.join('best_result', "pyg_pred_session1_" + args.model + "_" + str(model_id) + ".json")
    result_path = "../submit/submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".json"
    with open(result_path, 'w+') as f:
        for i in range(len(test_id)):
            y_dict = {}
            y_dict["item_id"] = int(test_id[i])
            y_dict["score"] = float(y_pred[i])
            json.dump(y_dict, f)
            f.write('\n')
