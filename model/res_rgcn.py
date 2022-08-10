import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, LayerNorm
from torch_geometric.nn import RGCNConv


class ResRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_bases, n_layers=3, dropout=0.4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.dropout = dropout
        # self.bn = BatchNorm1d(in_channels)
        # self.ln = LayerNorm(hidden_channels)
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        # self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases))
        for i in range(n_layers):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))
            # self.convs.append(BatchNorm1d(hidden_channels))
        # self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations, num_bases=num_bases))

    def forward(self, x, edge_index, edge_type):
        # x = self.bn(x)
        x = self.lin1(x)
        # x = self.ln(x)
        for i, conv in enumerate(self.convs):
            x_ = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x_ = x_.relu_()
                x = x + F.dropout(x_, p=self.dropout, training=self.training)
        
        x = self.lin2(x)
        return x