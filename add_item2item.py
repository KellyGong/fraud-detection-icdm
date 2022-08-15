
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
    dist.fill_diagonal_(-1) # remove diagonal
    dist = np.array(dist)
    dist[np.where(dist==0)] = -1 # remove nodes that have all 0 feature
    off_diag_ind = dist!=-1
    off_diag = dist[off_diag_ind]
    threshold = np.partition(off_diag, num_edge2add)[:num_edge2add].max()
    edge2add_idx = np.where((dist<=threshold) & off_diag_ind)
    sources, targets = all_sample_idx[edge2add_idx[0]], all_sample_idx[edge2add_idx[1]]
    hgraph['item', 'I', 'item'].edge_index = torch.vstack([sources, targets])
    print("It took {} minutes to add {} edges among {} nodes".format((time.time() - start_time)/60, num_edge2add, torch.numel(all_sample_idx)))
    return hgraph

hgraph = add_item_item_edge(hgraph, args.node_sample, args.edge_add, train_idx, val_idx, test_idx)