from dataclasses import dataclass
import os
import os.path as osp
import argparse
import json
from tabnanny import verbose
import time

from utils import EarlyStop, setup_seed

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import permutations
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborLoader, HGTLoader
import torch_geometric.transforms as T
from sklearn.metrics import average_precision_score

from torch.nn import Linear
from torch_geometric.nn import RGCNConv

from model import RGCN, RGPRGNN, RGAT, Node_Transformation, HGT, ResRGCN, Post_Transformation
import nni
import wandb
import random
from info_nce import InfoNCE
from losses import SupConLoss


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dataset/pyg_data/icdm2022_session2.pt')
# parser.add_argument('--dataset', type=str, default='dataset/pyg_data/icdm2022_session1_debug.pt')
parser.add_argument('--labeled-class', type=str, default='item')
parser.add_argument("--batch_size", type=int, default=64,
                    help="Mini-batch size. If -1, use full graph training.")
parser.add_argument("--model", choices=["RGCN", "RGPRGNN", "RGAT", "HGT", "ResRGCN"], default="RGPRGNN")

parser.add_argument("--fanout", type=int, default=150,
                    help="Fan-out of neighbor sampling.")
parser.add_argument("--n_layers", type=int, default=3,
                    help="number of propagation rounds")

parser.add_argument("--test-file", type=str, default="dataset/icdm2022_session2_test_ids.txt")
parser.add_argument("--json-file", type=str, default="pyg_pred_session2.json")

parser.add_argument("--dropedge", type=float, default=0.2)

# build item-item relation through feature proximity and metapath (common neighbor b)
parser.add_argument("--item_item", action='store_true', default=False)
parser.add_argument("--node_sample", type=int, default=80000)
parser.add_argument("--edge_add", type=int, default=500000)
parser.add_argument("--metapath", type=bool, default=False)
parser.add_argument("--meta_fraction", type=float, default=0.1)

parser.add_argument("--pre_transform", action='store_true', default=False)

parser.add_argument("--device", type=str, default="cuda")

args = parser.parse_args()


# model_id = random.randint(0, 100000)
# model_path = osp.join('best_model', args.model + "_" + str(model_id) + ".pth")

model_path = 'best_model/RGPRGNN_7235.pth'
model_id = '7235'

print(model_path)

setup_seed(2022)

device = torch.device(f'{args.device}' if torch.cuda.is_available() else 'cpu')
hgraph = torch.load(args.dataset)

labeled_class = args.labeled_class

# print(hgraph.node_types)

# if args.inference == False:
#     # train_idx = hgraph[labeled_class].pop('train_idx')
#     train_idx = hgraph[labeled_class]['train_idx']
#     if args.validation:
#         # val_idx = hgraph[labeled_class].pop('val_idx')
#         val_idx = hgraph[labeled_class]['val_idx']

test_id = [int(x) for x in open(args.test_file).readlines()]
converted_test_id = []
for i in test_id:
    converted_test_id.append(hgraph['item'].maps[i])
 
test_idx = torch.LongTensor(converted_test_id)

# C class balance parameter

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
    dist = torch.cdist(hgraph['item'].x[all_sample_idx], hgraph['item'].x[all_sample_idx])
    # dist.fill_diagonal_(-1) # remove diagonal
    dist[torch.triu(dist, diagonal=1)==0] = -1 # remove lower triangle
    dist[dist==0] = -1 # remove nodes that have all 0 feature
    dist = np.array(dist)
    # dist[np.where(dist==0)] = -1 # remove nodes that have all 0 feature
    off_diag_ind = dist!=-1
    off_diag = dist[off_diag_ind]
    threshold = np.partition(off_diag, num_edge2add//2)[:num_edge2add//2].max()
    edge2add_idx = np.where((dist<=threshold) & off_diag_ind)
    sources, targets = all_sample_idx[edge2add_idx[0]], all_sample_idx[edge2add_idx[1]]
    # sources = torch.cat([sources, targets])
    # targets = torch.cat([targets, sources])
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


model = torch.load(model_path, map_location=args.device)


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

        y_hat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))[
                start:start + batch_size]

        pbar.update(batch_size)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
    pbar.close()

    return torch.hstack(y_pred)




model = torch.load(model_path, map_location=args.device)
print(model_path)
y_pred = test()
result_path = osp.join('best_result', "pyg_pred_session2_" + args.model + "_" + str(model_id) + ".json")
with open(result_path, 'w+') as f:
    for i in range(len(test_id)):
        y_dict = {}
        y_dict["item_id"] = int(test_id[i])
        y_dict["score"] = float(y_pred[i])
        json.dump(y_dict, f)
        f.write('\n')
