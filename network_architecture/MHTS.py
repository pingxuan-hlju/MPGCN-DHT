from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from .GCN import GraphConvLayer
from .HTrans import Transformer
from .CPTrans import CPT
from .CSAT import CSAT
from torch.nn import Parameter


class MHTS(nn.Module):
    def __init__(self, num_layers, t_layers, num_heads, depth, height, width, node_num, in_features, dropout, edge_num_types):
        super().__init__()
        self.num_layers = num_layers
        self.eT_layers = t_layers
        self.aT_layers = t_layers
        self.num_heads = num_heads
        self.depth = depth
        self.height = height
        self.width = width
        self.node_num = node_num  # 长宽高乘积为结点数量
        self.in_features = in_features
        self.out_features = in_features
        self.node_dim = in_features  # 表示输入和输出张量的特征维度。在编码器层中，输入和输出的张量维度应该是相同的。

        self.dropout = dropout  # 表示在Transformer模型中应用的Dropout层的丢弃率,较大的丢弃率意味着保留的神经元更少，会导致更多的信息丢失。。
        self.edge_num_types = edge_num_types
        # 生成动态边的编码
        self.edge_encoder1 = nn.Embedding(edge_num_types, in_features)
        self.edge_encoder2 = nn.Embedding(node_num, edge_num_types)
        self.embedding_vectors1 = self.edge_encoder1.weight
        self.embedding_vectors2 = self.edge_encoder2.weight

        self.atlayers = nn.ModuleList()

        self.gconvs = nn.ModuleList()
        self.csat = nn.ModuleList()
        
        self.CPT3d = nn.ModuleList()
        self.line = nn.Linear(self.node_dim * 2, self.node_dim)
        self.lns1 = nn.ModuleList()
        self.bns1 = nn.ModuleList()
        self.bns2 = nn.ModuleList()
        self.etlayers1 = nn.ModuleList()
        self.HTrans = nn.ModuleList()
        self.conv1 = nn.Conv3d(640, 480, kernel_size=1)
        self.conv2 = nn.Conv3d(480, 320, kernel_size=1)
        self.conv = nn.Conv2d(2, 1, kernel_size=1)
        self.gamma1 = Parameter(torch.ones(1) * 0)
        self.gamma2 = Parameter(torch.ones(1) * 0.5)
        
        self.conv_out = nn.Conv3d(640, 320, kernel_size=1)

        for i in range(num_layers):
            
            self.CPT3d.append(CPT(self.depth, self.height, self.width, self.node_num, self.in_features))  # CP-Trans
            self.gconvs.append(GraphConvLayer(self.in_features, self.out_features))
            self.lns1.append(nn.LayerNorm(in_features, eps=1e-5))
            
            self.csat.append(CSAT(self.depth, self.height, self.width, self.node_num, self.in_features))
           
            self.HTrans.append(
                Transformer(self.node_dim, self.node_num, self.num_heads, self.eT_layers, self.edge_num_types,
                            self.embedding_vectors1, self.dropout,self.depth, self.height, self.width, self.in_features))

    def forward(self, data) -> Tensor:

        data = data.transpose(1, 2)  # [1,320,196] -> [1,196,320]
        out = data  # [1,196,320]
        rigion = data
        
        # 计算负向量差的指数
        # adjacency_matrix_exp = torch.exp(-torch.cdist(data, data))
        # 将邻接矩阵进行归一化处理
        # 使用输入数据的第三维作为向量进行点积生成邻接矩阵
        adjacency_matrix = torch.matmul(data, data.transpose(1, 2))
        # 将邻接矩阵进行归一化处理
        adjacency_matrix = torch.softmax(adjacency_matrix, dim=2)
        # 循环遍历每个样本
        # for b in range(data.shape[0]):
        #     # 获取当前样本的张量
        #     current_tensor = data[b]
        #     # 将张量重塑为形状 196 x 320
        #     reshaped_tensor = current_tensor.view(196, -1)
        #     # 计算余弦相似度矩阵
        #     cosine_sim = F.cosine_similarity(reshaped_tensor.unsqueeze(1), reshaped_tensor.unsqueeze(0), dim=2)
        #     # 将结果存入结果张量
        #     adjacency_matrix_cos[b] = cosine_sim

        for i in range(self.num_layers):
            # 第一条分支CP-trans
            data1 = out.transpose(1, 2)  # [1,196,320] -> [1,320,196]
            data1 = data1.reshape(-1, self.in_features, self.depth, self.height, self.width)
            data1 = self.CPT3d[i](data1)
            data1 = data1.reshape(-1, self.in_features, self.node_num)
            data1 = data1.transpose(1, 2)
            out1 = out + data1  # [1,196,320]

            # # 第二条分支MGCN
            # 多层面注意力
            data2 = out.transpose(1, 2)  # [1,196,320] -> [1,320,196]
            data2_a = data2.reshape(-1, self.in_features, self.depth, self.height, self.width)
            data2_a = self.csat[i](data2_a)
            data2_a = data2_a.reshape(-1, self.in_features, self.node_num).transpose(1, 2) # [1,320,196] ->[1,196,320]
            data2 = self.gconvs[i](data2_a, adjacency_matrix)
            out2 = out + data2 # [1,196,320]
            out_2 = self.lns1[i](out2)  # [1,196,320]
            
            # # 第二条分支边Htrans
            data2 = self.HTrans[i](out_2)
            out2 = out2 + data2  # (-1,196,320)

            # 输出单分支
            self.gamma1.data = torch.sigmoid(self.gamma1.data)
            # 输出双分支
            # out = (self.gamma1 * out1)   + out2
            out1 = out1.transpose(1, 2)
            out1 = out1.reshape(-1, self.in_features, self.depth, self.height, self.width)
            out2 = out2.transpose(1, 2)
            out2 = out2.reshape(-1, self.in_features, self.depth, self.height, self.width)
            out = torch.cat((out1,out2), dim=1)
            out = self.conv_out(out)
            
            out = out.reshape(-1, self.in_features, self.node_num).transpose(1, 2) # [1,320,196] ->[1,196,320]
            
        return out.transpose(1, 2)
        # return last_images.transpose(1, 2)



