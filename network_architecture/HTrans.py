from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F



class Transformer(nn.Module):
    def __init__(self, input_dim, node_num, num_heads, num_layers, edge_num_types,embedding_vectors,dropout, depth, height, width, in_features):
        super(Transformer, self).__init__()
        
        # print(embedding_vectors.size())  # torch.Size([196, 2])
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(input_dim, num_heads, edge_num_types, embedding_vectors,dropout, depth, height, width, node_num, in_features)
            for _ in range(num_layers)
        ])


    def forward(self, x):

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, edge_num_types, embedding_vectors, dropout, depth, height, width, node_num, in_features):
        super(EncoderLayer, self).__init__()
        # self.Line = nn.Linear(input_dim+edge_num_types, input_dim)
        self.depth = depth
        self.height = height
        self.width = width
        self.in_features = in_features
        self.node_num = node_num
        
        self.self_attention = MultiHeadAttention(input_dim, num_heads, edge_num_types, embedding_vectors, node_num,in_features)
        # self.feed_forward = FeedForward(self.in_features) # 卷积传
        self.feed_forward = FeedForward(input_dim)  # mlp传
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm1(x)
        residual = x
        x = self.feed_forward(x)  # mlp传
        x = x + residual
        # x = self.norm2(x)  # 去掉多余归一化
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads,  edge_num_types, embedding_vectors, node_num,in_features):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.embedding_vectors = embedding_vectors
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.mlp = nn.Linear(in_features+edge_num_types, in_features)
        # self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim, input_dim)
        self.node_num = node_num
        self.channel_embedding = nn.Parameter(torch.zeros(1, self.node_num, 1))  # 1*196*1

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        x = x + self.channel_embedding # 加上位置编码
         # 将两个张量在第二个维度上进行拼接
        self.DH = torch.matmul(x, self.embedding_vectors.T) # 生成超边邻接矩阵  [3, 196, 320]（320*8）  =[3, 196, 8]
        x_E = torch.cat((x, self.DH), dim=2)  # [3, 196, 320+8]

        query = self.mlp(x_E)
        key = self.mlp(x_E)
        value = self.mlp(x_E)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        scores = torch.matmul(query, key.transpose(2, 3))
        scores = scores / (self.head_dim ** 0.5)

        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)

        attended_values = attended_values.view(batch_size, seq_len, -1)

        # output = self.output_linear(attended_values) 去掉多余mlp
        output = attended_values

        return output



class FeedForward(nn.Module):
    def __init__(self, input_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim*2)
        self.linear2 = nn.Linear(input_dim*2, input_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)

        return x