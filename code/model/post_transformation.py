import torch
import torch.nn as nn
import torch.nn.functional as F


class Post_Transformation(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # self.in_dim = hidden_channels
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)

    def reset_parameters(self):
        # for transformation in self.transformations:
        #     torch.nn.init.xavier_normal_(transformation.weight)
        # torch.nn.init.xavier_normal_(self.node_embedding.weight)
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, x):
        # output = torch.zeros((x.shape[0], self.out_dim), device=x.device)
        # for node_i in range(self.n):
        #     node_index = torch.where(node_type==node_i)
        #     transformed_features = self.transformations[node_i](x[node_index])
        #     output[node_index] = transformed_features
        x = self.linear1(x)
        x = x.relu_()
        x = self.linear2(x)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x
