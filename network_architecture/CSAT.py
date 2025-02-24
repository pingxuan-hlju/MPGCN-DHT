import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class CCAM(nn.Module):
    def __init__(self, depth, height, width, node_num, in_features):
        super(CCAM, self).__init__()

        # 第一个分支：算通道之间的注意力
        self.line_c = nn.Linear(node_num, 16)
        self.C = Parameter(torch.ones(16, 1))

        # 第二个分支：算冠状面之间的注意力
        self.line_coronal = nn.Linear(in_features * height * width, 16)
        self.Coronal = Parameter(torch.ones(16, 1))

        # 第三个分支：横断面之间的注意力
        self.line_transverse = nn.Linear(in_features * depth * width, 16)
        self.Transverse = Parameter(torch.ones(16, 1))

        # 第四个分支：矢状面之间的注意力sagittal
        self.line_sagittal = nn.Linear(in_features * depth * height, 16)
        self.Sagittal = Parameter(torch.ones(16, 1))

    def forward(self, x):
        # 第一个分支：算通道之间的注意力
        x_c = x.reshape(x.size(0), x.size(1), -1)  # torch.Size([2, 320, 196])
        x_c = self.line_c(x_c)  # torch.Size([2, 320, 196])
        x_c = F.tanh(x_c)  #
        x_c_1 = x_c @ self.C  # [2, 320, 196] * [196, 1] torch.Size([2, 320, 1])
        x_c_1 = x_c_1.unsqueeze(-1).unsqueeze(-1)  # torch.Size([2, 320, 1, 1, 1])

        # 算冠状面之间的注意力  (2, 320, 4, 7, 7)
        x_coronal = x.transpose(1, 2).reshape(x.size(0), x.size(2), -1)  # torch.Size([2, 4, 15680])
        x_coronal = self.line_coronal(x_coronal)  # torch.Size([2, 4, 15680])
        x_coronal = F.tanh(x_coronal)  # torch.Size([2, 4, 15680])
        x_coronal_1 = x_coronal @ self.Coronal  # torch.Size([2, 4, 15680])@torch.Size([15680, 1]) -》 torch.Size([2, 4, 1])
        x_coronal_1 = x_coronal_1.unsqueeze(1).unsqueeze(-1)  # rch.Size([2, 1, 4, 1, 1])

        # 第三个分支：横断面之间的注意力 (2, 320, 4, 7, 7)
        x_transverse = x.transpose(1, 3).reshape(x.size(0), x.size(3), -1)  # torch.Size([2, 7, 8960])
        x_transverse = self.line_transverse(x_transverse)  # torch.Size([7, 8960])
        x_transverse = F.tanh(x_transverse)  # torch.Size([7, 8960])
        x_transverse_1 = x_transverse @ self.Transverse  # torch.Size([2, 7, 8960])@torch.Size([8960, 1]) -》 torch.Size([2, 7, 1])
        x_transverse_1 = x_transverse_1.unsqueeze(1).unsqueeze(2)  # torch.Size([2, 1, 1, 7, 1])

        # 第四个分支：矢状面之间的注意力
        x_sagittal = x.transpose(1, 4).reshape(x.size(0), x.size(4), -1)  # torch.Size([2, 7, 8960])
        x_sagittal = self.line_sagittal(x_sagittal)  # torch.Size([7, 8960])
        x_sagittal = F.tanh(x_sagittal)  # torch.Size([7, 8960])
        x_sagittal_1 = x_sagittal @ self.Sagittal  # torch.Size([7, 8960])@torch.Size([8960, 1]) -》 torch.Size([7, 1])
        x_sagittal_1 = x_sagittal_1.transpose(-1, -2).unsqueeze(1).unsqueeze(2)  # torch.Size([2, 1, 1, 1, 7])

        output = x_c_1 * x_coronal_1 * x_transverse_1 * x_sagittal_1
        output = F.sigmoid(output)

        return output


class LSAM(nn.Module):
    def __init__(self, depth, height, width, node_num, in_features):
        super(LSAM, self).__init__()
        self.channel = in_features
        self.node_num = node_num
        self.depth = depth
        self.height = height
        self.width = width
        self.mlp = nn.Linear(in_features * 8, 1)
        self.conv = nn.Conv3d(in_features, 1, kernel_size=2, stride=2)

        self.conv_cat = nn.Conv3d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        _, _, r_depth, r_height, r_width = x.size()
        # print(x.shape,'x.size')
        if r_depth % 2 == 1:
            padding = (0, 0, 0, 0, 1,0)  # torch.Size([2, 320, 3, 5, 7]) -> torch.Size([2, 320, 4, 5, 7])  (left, right, top, bottom, front, back)
            x = F.pad(x, padding)
            # print(x.shape, 'x.size')
        if r_height % 2 == 1:
            padding = (0, 0, 1, 0, 0,0)  # torch.Size([2, 320, 4, 5, 7]) -> torch.Size([2, 320, 4, 6, 7])  (left, right, top, bottom, front, back)
            x = F.pad(x, padding)
            # print(x.shape, 'x.size')
        if r_width % 2 == 1:
            padding = (1, 0, 0, 0, 0,0)  # torch.Size([2, 320, 4, 6, 7]) -> torch.Size([2, 320, 4, 6, 8])  (left, right, top, bottom, front, back)
            x = F.pad(x, padding)
            # print(x.shape, 'x.size')
        blocks = x.unfold(2, 2, 2).unfold(3, 2, 2).unfold(4, 2, 2)  # torch.Size([2, 320, 2, 4, 4, 2, 2, 2])
        blocks = blocks.permute(0, 1, 5, 6, 7, 2, 3, 4)  # torch.Size([2, 320, 2, 2, 2, 2, 4, 4])
        bs, _, _, _, _, depth, height, width = blocks.size()
        blocks = blocks.reshape(bs, self.channel, 2, 2, 2, -1)  # torch.Size([2, 320, 2, 2, 2, 32])
        blocks = blocks.permute(0, 5, 1, 2, 3, 4)  # torch.Size([2, 32, 320, 2, 2, 2])

        # 线性池化
        line_input = blocks.reshape(bs, -1, self.channel * 8)  # torch.Size([2, 32, 2560])
        mlp_output = self.mlp(line_input)  # torch.Size([2, 32, 1])
        mlp_output = mlp_output.transpose(-1, -2).reshape(blocks.size(0), 1, depth, height,
                                                          width)  # torch.Size([2, 1, 2, 4, 4])

        # 卷积池化
        conv_input = blocks.reshape(-1, self.channel, 2, 2, 2)  # torch.Size([64, 320, 2, 2, 2])
        conv_output = self.conv(conv_input)  # torch.Size([64, 1, 1, 1, 1])
        conv_output = conv_output.reshape(bs, -1, 1).transpose(-1, -2).reshape(bs, 1, depth, height,
                                                                               width)  # torch.Size([2, 1, 2, 4, 4])

        concat_input = torch.cat((mlp_output, conv_output), dim=1)  # torch.Size([2, 2, 2, 4, 4])

        # 卷积
        concat_output = self.conv_cat(concat_input)  # torch.Size([2, 1, 2, 4, 4])

        output_tensor = F.interpolate(concat_output, size=(r_depth, r_height, r_width), mode='nearest')
        output_tensor = torch.sigmoid(output_tensor)
        return output_tensor


class CSAT(nn.Module):
    def __init__(self, depth, height, width, node_num, in_features):
        super(CSAT, self).__init__()
        # self.conv1 = nn.Conv3d(in_features, in_features, kernel_size=1)
        self.conv2 = nn.Conv3d(in_features, in_features, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv3d(in_features, in_features, kernel_size=1)
        self.ccam = CCAM(depth, height, width, node_num, in_features)
        self.lsam = LSAM(depth, height, width, node_num, in_features)

    def forward(self, x):
        begin = x
        # x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)

        c = self.ccam(x).cuda()  # torch.Size([2, 320, 4, 7, 7])
        l = self.lsam(x)  # torch.Size([2, 1, 4, 7, 7])

        att = F.sigmoid(c * l)  # 广播机制
        # print(att.shape)  # torch.Size([2, 320, 4, 7, 7])

        x = x * att
        out = begin + x  # 残差
        return out

# # # 创建模型实例
# depth = 3
# height = 5
# width = 7
# node_num = 105
# in_features = 320
# model = CSAT(depth, height, width, node_num, in_features).cuda()
# 
# # 创建输入张量
# input = torch.randn(2, 320, 3, 5, 7).cuda()
# 
# # 前向传播
# output = model(input)
# 
# print(output.size())  # 输出: torch.Size([2, 1, 4, 7, 7])