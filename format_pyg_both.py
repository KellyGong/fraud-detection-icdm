import argparse
from pydoc import describe
from tkinter import W
from torch_geometric.data import HeteroData
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import pickle as pkl

# edge_size = 0
# node_size = 0


def read_node_atts_both(node_file1, node_file2, pyg_file, label_file=None):
    node_maps = {}
    node_embeds = {}
    count = 0
    lack_num = {}
    node_counts = node_size1 + node_size2
    if osp.exists(pyg_file + ".nodes.pyg") == False:
        print("Start loading node information")
        process = tqdm(total=node_counts)
        with open(node_file1, 'r') as rf:
            while True:
                line = rf.readline()
                if line is None or len(line) == 0:
                    break
                info = line.strip().split(",")

                node_id = int(info[0])
                node_type = info[1].strip()

                node_maps.setdefault(node_type, {})
                node_id_v2 = len(node_maps[node_type])
                node_maps[node_type][node_id] = node_id_v2

                node_embeds.setdefault(node_type, {})
                lack_num.setdefault(node_type, 0)
                if node_type == 'item':
                    if len(info[2]) < 50:
                        node_embeds[node_type][node_id_v2] = np.zeros(256, dtype=np.float32)
                        lack_num[node_type] += 1
                    else:
                        node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)
                else:
                    if len(info[2]) < 50:
                        node_embeds[node_type][node_id_v2] = np.zeros(256, dtype=np.float32)
                        lack_num[node_type] += 1
                    else:
                        node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)

               
                count += 1
                process.update(1)

        with open(node_file2, 'r') as rf:
            while True:
                line = rf.readline()
                if line is None or len(line) == 0:
                    break
                info = line.strip().split(",")

                node_id = int(info[0]) + node_size1
                node_type = info[1].strip()

                # node_maps.setdefault(node_type, {})
                node_id_v2 = len(node_maps[node_type])
                node_maps[node_type][node_id] = node_id_v2

                # node_embeds.setdefault(node_type, {})
                # lack_num.setdefault(node_type, 0)
                if node_type == 'item':
                    if len(info[2]) < 50:
                        node_embeds[node_type][node_id_v2] = np.zeros(256, dtype=np.float32)
                        lack_num[node_type] += 1
                    else:
                        node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)
                else:
                    if len(info[2]) < 50:
                        node_embeds[node_type][node_id_v2] = np.zeros(256, dtype=np.float32)
                        lack_num[node_type] += 1
                    else:
                        node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)

                count += 1
                process.update(1)

        process.update(node_counts % 100000)
        process.close()
        print("Complete loading node information\n")

        print("Num of total nodes:", count)
        print('Node_types:', list(node_maps.keys()))
        print('Node_type Num Num_lack_feature:')
        for node_type in node_maps:
            print(node_type, len(node_maps[node_type]), lack_num[node_type])

        labels = []
        if label_file is not None:
            labels_info = [x.strip().split(",") for x in open(label_file).readlines()]
            for i in range(len(labels_info)):
                try:
                    x = labels_info[i]
                    item_id = node_maps['item'][int(x[0])]
                    label = int(x[1])
                    labels.append([item_id, label])
                except:
                    continue

        nodes_dict = {'maps': node_maps, 'embeds': node_embeds}
        nodes_dict['labels'] = {}
        nodes_dict['labels']['item'] = labels
        print('\n')
        print('Start saving pkl-style node information')
        pkl.dump(nodes_dict, open(pyg_file + ".nodes.pyg", 'wb'), pkl.HIGHEST_PROTOCOL)
        print('Complete saving pkl-style node information\n')

    else:
        nodes = pkl.load(open(pyg_file + ".nodes.pyg", 'rb'))
        node_embeds = nodes['embeds']
        node_maps = nodes['maps']
        labels = nodes['labels']['item']

    graph = HeteroData()

    print("Start converting into pyg data")
    for node_type in tqdm(node_embeds, desc="Node features, numbers and mapping"):
        graph[node_type].x = torch.empty(len(node_maps[node_type]), 256)
        # for nid, embedding in tqdm(node_embeds[node_type].items()):
        for nid, embedding in node_embeds[node_type].items():
            graph[node_type].x[nid] = torch.from_numpy(embedding)
        graph[node_type].num_nodes = len(node_maps[node_type])
        graph[node_type].maps = node_maps[node_type]

    if label_file is not None:
        graph['item'].y = torch.zeros(len(node_maps['item']), dtype=torch.long) - 1
        for index, label in tqdm(labels, desc="Node labels"):
            graph['item'].y[index] = label

        indices = (graph['item'].y != -1).nonzero().squeeze()
        print("Num of true labeled nodes:{}".format(indices.shape[0]))
        train_val_random = torch.randperm(indices.shape[0])
        train_idx = indices[train_val_random][:int(indices.shape[0] * 0.8)]
        val_idx = indices[train_val_random][int(indices.shape[0] * 0.8):]
        graph['item'].train_idx = train_idx
        graph['item'].val_idx = val_idx

    for ntype in graph.node_types:
        graph[ntype].n_id = torch.arange(graph[ntype].num_nodes)
    print("Complete converting into pyg data\n")

#    print("Start saving into pyg data")
#    torch.save(graph, pyg_file + ".pt")
#    print("Complete saving into pyg data\n")
    return graph


def format_pyg_graph_both(edge_file1, edge_file2, node_file1, node_file2, pyg_file, label_file=None):
    if osp.exists(pyg_file + ".pt") and args.reload == False:
#        graph = torch.load(pyg_file + ".pt")
        print("PyG graph of " + ("session2" if "session2" in pyg_file else "session1") + " has generated")
        return 0
    else:
        print("##########################################")
        print("### Start generating PyG graph of session1 and session2")
        print("##########################################\n")
        graph = read_node_atts_both(node_file1, node_file2, pyg_file, label_file)

    print("Start loading edge information")
    process = tqdm(total=edge_size1+edge_size2)
    edges = {}
    count = 0
    with open(edge_file1, 'r') as rf:
        while True:
            line = rf.readline()
            if line is None or len(line) == 0:
                break
            line_info = line.strip().split(",")
            process.update(1)
            source_id, dest_id, source_type, dest_type, edge_type = line_info
            source_id = graph[source_type].maps[int(source_id)]
            dest_id = graph[dest_type].maps[int(dest_id)]
            edges.setdefault(edge_type, {})
            edges[edge_type].setdefault('source', []).append(int(source_id))
            edges[edge_type].setdefault('dest', []).append(int(dest_id))
            edges[edge_type].setdefault('source_type', source_type)
            edges[edge_type].setdefault('dest_type', dest_type)
            count += 1
    with open(edge_file2, 'r') as rf:
        while True:
            line = rf.readline()
            if line is None or len(line) == 0:
                break
            line_info = line.strip().split(",")
            process.update(1)
            source_id, dest_id, source_type, dest_type, edge_type = line_info
            source_id = graph[source_type].maps[int(source_id)+node_size1]
            dest_id = graph[dest_type].maps[int(dest_id)+node_size1]
            # edges.setdefault(edge_type, {})
            edges[edge_type].setdefault('source', []).append(int(source_id))
            edges[edge_type].setdefault('dest', []).append(int(dest_id))
            edges[edge_type].setdefault('source_type', source_type)
            edges[edge_type].setdefault('dest_type', dest_type)
            count += 1
    process.update((edge_size1+edge_size2) % 100000)
    process.close()
    print(f'Edge Count: {str(count)}')
    print('Complete loading edge information\n')

    print('Start converting edge information')
    for edge_type in edges:
        source_type = edges[edge_type]['source_type']
        dest_type = edges[edge_type]['dest_type']
        source = torch.tensor(edges[edge_type]['source'], dtype=torch.long)
        dest = torch.tensor(edges[edge_type]['dest'], dtype=torch.long)
        graph[(source_type, edge_type, dest_type)].edge_index = torch.vstack([source, dest])

    for edge_type in [('b', 'A_1', 'item'),
                      ('f', 'B', 'item'),
                      ('a', 'G_1', 'f'),
                      ('f', 'G', 'a'),
                      ('a', 'H_1', 'e'),
                      ('f', 'C', 'd'),
                      ('f', 'D', 'c'),
                      ('c', 'D_1', 'f'),
                      ('f', 'F', 'e'),
                      ('item', 'B_1', 'f'),
                      ('item', 'A', 'b'),
                      ('e', 'F_1', 'f'),
                      ('e', 'H', 'a'),
                      ('d', 'C_1', 'f')]:
        try:
            temp = graph[edge_type].edge_index
            del graph[edge_type]
            graph[edge_type].edge_index = temp
        except:
            continue

    print('Complete converting edge information\n')
    print('Start saving into pyg data')
    torch.save(graph, pyg_file + ".pt")
    print('Complete saving into pyg data\n')

    print("##########################################")
    print("### Complete generating PyG graph of " + ("session2" if "session2" in args.storefile else "session1"))
    print("##########################################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph1', type=str, default="dataset/icdm2022_session1_edges.csv")
    parser.add_argument('--node1', type=str, default="dataset/icdm2022_session1_nodes.csv")
    parser.add_argument('--graph2', type=str, default="dataset/icdm2022_session2_edges.csv")
    parser.add_argument('--node2', type=str, default="dataset/icdm2022_session2_nodes.csv")
    parser.add_argument('--label', type=str, default="dataset/icdm2022_session1_train_labels.csv")
    parser.add_argument('--storefile', type=str, default="dataset/pyg_data/icdm2022_session_both")
    parser.add_argument('--reload', type=bool, default=False, help="Whether node features should be reloaded")
    args = parser.parse_args()
    edge_size2 = 120691444
    node_size2 = 10284026

    edge_size1 = 157814864
    node_size1 = 13806619
    if args.graph1 is not None and args.graph2 is not None and args.storefile is not None and args.node1 is not None and args.node2 is not None:
        format_pyg_graph_both(args.graph1, args.graph2, args.node1, args.node2, args.storefile, args.label)
        # read_node_atts(args.node, args.storefile, args.label)
