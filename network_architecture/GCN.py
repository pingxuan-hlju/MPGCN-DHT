from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency_matrix):
        # device = torch.cuda()  # device = torch.device('cuda:0')
        adjacency_matrix = adjacency_matrix.cuda()
        x = x.cuda()
        x = x.float()  # 将 x 转换为单精度数据类型
        x = torch.matmul(adjacency_matrix, x)
        x = self.linear(x)
        x = torch.relu(x)
        return x
