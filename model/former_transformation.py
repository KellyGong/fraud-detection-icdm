import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


# class Node_Transformation(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_transformation):
#         super().__init__()
#         self.n = num_transformation
#         self.out_dim = hidden_channels
#         self.transformations = torch.nn.ModuleList()
#         for i in range(num_transformation):
#             self.transformations.append(Linear(in_channels, hidden_channels))
#         self.reset_parameters()

#     def reset_parameters(self):
#         for transformation in self.transformations:
#             torch.nn.init.xavier_normal_(transformation.weight)

#     def forward(self, x, node_type):
#         output = torch.zeros((x.shape[0], self.out_dim), device=x.device)
#         for node_i in range(self.n):
#             node_index = torch.where(node_type==node_i)
#             transformed_features = self.transformations[node_i](x[node_index])
#             output[node_index] = transformed_features
#         return output


# class Node_Transformation(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_transformation):
#         super().__init__()
#         self.n = num_transformation
#         self.out_dim = hidden_channels
#         self.transformations = torch.nn.ModuleList()
#         for i in range(num_transformation):
#             self.transformations.append(Linear(in_channels, hidden_channels))
#         self.reset_parameters()

class Node_Transformation(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_transformation):
        super().__init__()
        self.n = num_transformation
        self.out_dim = hidden_channels
        self.node_embedding = nn.Embedding(self.n, hidden_channels)
        self.linear = nn.Linear(in_channels, hidden_channels)

    def reset_parameters(self):
        # for transformation in self.transformations:
        #     torch.nn.init.xavier_normal_(transformation.weight)
        torch.nn.init.xavier_normal_(self.node_embedding.weight)
        torch.nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x, node_type, item_id):
        # output = torch.zeros((x.shape[0], self.out_dim), device=x.device)
        # for node_i in range(self.n):
        #     node_index = torch.where(node_type==node_i)
        #     transformed_features = self.transformations[node_i](x[node_index])
        #     output[node_index] = transformed_features
        output = self.node_embedding(node_type)
        node_index = torch.where(node_type==item_id)
        output[node_index] = self.linear(x[node_index])
        return output
