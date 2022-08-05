import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import RGCNConv


class RGPRGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_bases, alpha=0.1, n_layers=3, dropout=0.4, activation='relu', pre_transform=False):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))

        self.pre_transform = pre_transform
        # GPRGNN
        if not self.pre_transform:
            self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        K = n_layers
        self.K = K
        self.alpha = alpha
        self.dropout = dropout

        TEMP = alpha*(1-alpha)**np.arange(K+1)
        TEMP[-1] = (1-alpha)**K

        self.temp = Parameter(torch.tensor(TEMP))
        self.reset_parameters()

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaklyrelu':
            self.activation = F.leaky_relu
        elif activation == 'elu':
            self.activation = F.elu

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_type):
        # x = self.lin1(x)
        # hidden = x*(self.temp[0])
        # x = hidden
        # for i, conv in enumerate(self.convs):
        #     x = conv(x, edge_index, edge_type)
        #     if i < len(self.convs) - 1:
        #         x = x.relu_()
        #         x = F.dropout(x, p=0.4, training=self.training)
        #     hidden = hidden + self.temp[i+1]*x
        
        # hidden = self.lin2(hidden)

        if not self.pre_transform:
            x = self.lin1(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        hidden = x*(self.temp[0])
        x = hidden
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                # x = x / torch.norm(x, dim=1, keepdim=True)
            hidden = hidden + self.temp[i+1]*x
        
        # hidden = hidden / torch.norm(hidden, dim=1, keepdim=True)
        hidden = self.lin2(hidden)
        return hidden