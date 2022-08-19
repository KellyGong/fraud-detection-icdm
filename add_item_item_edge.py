from itertools import permutations
from torch_geometric.utils import to_undirected
# build item-item relation
parser.add_argument("--item_item", action='store_true', default=False)
parser.add_argument("--node_sample", type=int, default=80000)
parser.add_argument("--edge_add", type=int, default=500000)
parser.add_argument("--metapath", type=bool, default=True)
parser.add_argument("--meta_fraction", type=float, default=0.1)
if args.metapath:
    num_relations += 1
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

if args.metapath:
    batch = add_metapath(batch)
