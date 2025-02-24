import functools
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair,_triple
class Attention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 proj_drop,
                 node_num,
                 kernel_size=2,  # 自己调参
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True,
                 channel_attention =False,
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop
        self.channel_attention = channel_attention
        self.conv_q = nn.Conv3d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias, groups=channels)
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k = nn.Conv3d(channels, channels, kernel_size, stride_kv, padding_kv, bias=attention_bias, groups=channels)
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v = nn.Conv3d(channels, channels, kernel_size, stride_kv, padding_kv, bias=attention_bias, groups=channels)
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)

        self.attention1 = nn.MultiheadAttention(embed_dim=channels,
                                               bias=attention_bias,
                                               batch_first=True,
                                               # dropout = 0.0,
                                               num_heads=self.num_heads)  # num_heads=1)
        self.attention2 = nn.MultiheadAttention(embed_dim=node_num,
                                               bias=attention_bias,
                                               batch_first=True,
                                               # dropout = 0.0,
                                               num_heads=self.num_heads)  # num_heads=1)

        self.positional_embedding = nn.Parameter(torch.zeros(1, node_num, 1))  # 1*196*1
        self.channel_embedding = nn.Parameter(torch.zeros(1, channels, 1))  # 1*320*1


    def _build_projection(self, x, qkv):


        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 4, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 4, 1, 2, 3)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 4, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 4, 1, 2, 3)
        elif qkv == "v":
            # x1 = F.relu(self.conv_v(x))
            # x1 = x1.permute(0, 2, 3, 4, 1)
            # x1 = self.layernorm_v(x1)
            # proj = x1.permute(0, 4, 1, 2, 3)
            proj = x

        return proj

    def forward_conv(self, x):
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")

        return q, k, v

    def forward(self, x):
        if self.channel_attention == False:
            x1 = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3] * x.shape[4]).transpose(1, 2) + self.positional_embedding
            x = x1.transpose(1,2).reshape(x.shape[0], x.shape[1], x.shape[2] , x.shape[3] , x.shape[4])  # torch.Size([2, 320, 4, 7, 7])
        else:
            x2 = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3] * x.shape[4]) + self.channel_embedding
            x = x2.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        q, k, v = self.forward_conv(x)
        q = q.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3] * x.shape[4])
        k = k.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3] * x.shape[4])
        v = v.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3] * x.shape[4])
        if self.channel_attention == False:
            q = q.permute(0, 2, 1)  # torch.Size([2, 196, 320])
            k = k.permute(0, 2, 1)
            v = v.permute(0, 2, 1)
        if self.channel_attention == False:
            # print(q.shape, 'q', v.shape, 'v', k.shape, 'k')
            x1 = self.attention1(query=q, value=v, key=k, need_weights=False)  # torch.Size([2, 196, 320])

        else:
            # print(q.shape,'q',v.shape,'v',k.shape,'k')
            x1 = self.attention2(query=q, value=v, key=k, need_weights=False)


        if self.channel_attention == False:
            x1 = x1[0].permute(0, 2, 1)  # [0]输出张量 [1]权重矩阵
        else:
            x1 = x1[0]
            # x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        x1 = x1.view(x1.shape[0], x1.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x1 = F.dropout(x1, self.proj_drop)

        return x1


class Transformer(nn.Module):

    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads,
                 dpr,
                 node_num,
                 proj_drop=0.1,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1,
                 ):
        super().__init__()

        self.attention_output1 = Attention(channels=out_channels,
                                          num_heads=num_heads,
                                          proj_drop=proj_drop,
                                          node_num=node_num,
                                          padding_q=padding_q,
                                          padding_kv=padding_kv,
                                          stride_kv=stride_kv,
                                          stride_q=stride_q,
                                          attention_bias=attention_bias,
                                           channel_attention=False  # 算位置之间的注意力
                                          )
        self.attention_output2 = Attention(channels=out_channels,
                                          num_heads=num_heads,
                                          proj_drop=proj_drop,
                                          node_num=node_num,
                                          padding_q=padding_q,
                                          padding_kv=padding_kv,
                                          stride_kv=stride_kv,
                                          stride_q=stride_q,
                                          attention_bias=attention_bias,
                                           channel_attention=True  # 算通道之间的注意力
                                          )


        self.conv1 = nn.Conv3d(out_channels, out_channels, 2, 1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus = Wide_Focus(out_channels, out_channels)
        self.gamma3 = Parameter(torch.ones(1) * 0.5)
        self.gamma4 = Parameter(torch.ones(1) * 0.5)


    def forward(self, x):
        # 残差在这里加

        x1 = self.attention_output1(x)  # torch.Size([2, 320, 4, 7, 7])
        x2 = self.attention_output2(x)  # torch.Size([2, 320, 4, 7, 7])
        # self.gamma3.data = torch.sigmoid(self.gamma3.data)
        # self.gamma4.data = torch.sigmoid(self.gamma4.data)
        x1 = (self.gamma3 * x1) + (self.gamma4 * x2)
        # x2 = self
        x1 = self.conv1(x1)
        x2 = torch.add(x1, x)
        x3 = x2.permute(0, 2, 3, 4, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 4, 1, 2, 3)
        x3 = self.wide_focus(x3)
        x3 = torch.add(x2, x3)
        return x3

class LightUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightUNet, self).__init__()

        # 编码器部分
        self.encoder1 = self.build_encoder_block(in_channels, in_channels*2)
        self.encoder2 = self.build_encoder_block(in_channels*2, in_channels*4)

        # 解码器部分
        self.decoder2 = self.build_decoder_block(in_channels*4, in_channels*2)
        self.decoder1 = self.build_decoder_block(in_channels*4, in_channels)

        self.conv = nn.Conv3d(in_channels*2,in_channels,kernel_size=3,padding=1)

        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3)  # depth=4
        self.conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=3)  # depth=8

    def forward(self, x):
        bs, channels, depth, height, width = x.size()  #  [2, 320, 8, 14, 14]
        if height == 7:
            padding = (1, 0, 1, 0, 0, 0)  # torch.Size([2, 320, 4, 7, 7]) -> torch.Size([2, 320, 4, 8, 8])  (left, right, top, bottom, front, back)
            x = F.pad(x, padding)
        # 编码器部分
        x1 = self.encoder1(x)  # 2, 640, 2, 4, 4  [2, 320, 8, 14, 14]->[2, 640, 4, 7, 7]
        if height == 14:
            padding = (1, 0, 1, 0, 0, 0)  # torch.Size([2, 320, 4, 7, 7]) -> torch.Size([2, 320, 4, 8, 8])  (left, right, top, bottom, front, back)
            x1 = F.pad(x1, padding)  # ->[2, 640, 4, 7, 7] ->[2, 640, 4, 8, 8]  
        x2 = self.encoder2(x1)  # 2, 1280, 1, 2, 2   [2, 1280, 2, 4, 4]
        # 解码器部分
        x3 = self.decoder2(x2)  # 2, 640, 2, 4, 4    [2, 640, 4, 8, 8]
        x3 = torch.cat((x3,x1), dim=1)  # 2, 1280, 2, 4, 4  [2, 1280, 4, 8, 8]
        x4 = self.decoder1(x3)   # 2, 320, 4, 8, 8  [2, 320, 8, 16, 16]
        if height == 14:
            padding = (0, 0, 0, 0, 1, 1)  # torch.Size([2, 320, 10, 16, 16]) -> torch.Size([2, 320, 8, 14, 14])  (left, right, top, bottom, front, back)
            x4 = F.pad(x4, padding)
            x4 = self.conv3(x4)
        x4 = torch.cat((x4,x), dim=1)  # 2, 640, 4, 8, 8
        x = self.conv(x4)
        if height == 7:
            padding = (1, 0, 1, 0, 1, 1)  # torch.Size([2, 320, 4, 8, 8]) -> torch.Size([2, 320, 6, 9, 9])  (left, right, top, bottom, front, back)
            x = F.pad(x, padding)
            x = self.conv2(x)

        return x

    def build_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def build_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        )
class DDConv_3D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        super(DDConv_3D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ConstantPad3d(padding,value=0)
        self.conv = SConv3D(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = SConv3D(inc, 3*kernel_size*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)  # 共有3*3*3个位置，每个位置3个值
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = SConv3D(inc, kernel_size*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)   # 计算所有位置偏置对应的3*3*3*3=81的偏置值
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))  # 计算kernel的权重

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 3   # 通道//3

        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype).to("cuda")  # 将学习得到的偏置加上原始的得到偏移后的位置
        # 这部分代码对 p 张量进行处理，生成了四个张量 q_lt、q_rb、q_lb 和 q_rt。这些张量都具有形状 (batch_size, H, W, D, N)，
        # 其中 batch_size 是批量大小，H、W、D 是 x 张量的高度、宽度和深度，N 是 offset 张量的通道数除以 3
        # 这些张量表示了坐标网格中的四个角的坐标。q_lt 表示左上角，q_rb 表示右下角，q_lb 表示左下角，q_rt 表示右上角。
        # 通过使用 torch.clamp 函数，确保坐标值在合理的范围内，避免越界。
        p = p.contiguous().permute(0, 2, 3, 4, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:2*N], 0, x.size(3)-1),torch.clamp(p[..., 2*N:], 0, x.size(4)-1)], dim=-1)
        # 这部分代码计算了四个角点的权重值，使用的是坐标网格和 p 张量的差值。通过 (1 + (q[..., :N].type_as(p) - p[..., :N])) 的形式，计算了每个角点的权重，
        # 其中 (q[..., :N].type_as(p) - p[..., :N])
        # 表示坐标网格和 p 张量在 x 坐标上的差值，然后加上 1。这样可以得到一个权重张量，表示每个角点在插值过程中的贡献程度。
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:2*N].type_as(p) - p[..., N:2*N])) *(1 + (q_lt[..., 2*N:].type_as(p) - p[..., 2*N:]))
        g_rb =  (1 + (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rb[..., N:2*N].type_as(p) - p[..., N:2*N])) *(1 + (q_rb[..., 2*N:].type_as(p) - p[..., 2*N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lb[..., N:2*N].type_as(p) - p[..., N:2*N])) *(1 + (q_lb[..., 2*N:].type_as(p) - p[..., 2*N:]))
        g_rt = (1 + (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:2*N].type_as(p) - p[..., N:2*N])) *(1 + (q_rt[..., 2*N:].type_as(p) - p[..., 2*N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 4, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)  # torch.Size([2, 320, 24, 42, 42])
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):  # N为偏置的通道数//3   为9  对应一个位置为中心点
        p_n_x, p_n_y, p_n_z= torch.meshgrid(  # 该函数用于生成偏移量 p_n。xyz三个方向的偏移
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),  # 都是生成27个点，移动反向不同
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),  #
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))  #
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y),torch.flatten(p_n_z)], 0)  #【3*3*3】*3
        p_n = p_n.view(1, 3*N, 1, 1,1).type(dtype)

        return p_n  # torch.Size([1, 81, 1, 1, 1])

    def _get_p_0(self, h, w,d, N, dtype):
        #该函数用于生成基准偏移量 p_0。 对应全部位置
        p_0_x, p_0_y,p_0_z = torch.meshgrid(   # torch.meshgrid 函数通常用于生成一对坐标网格，用于计算网格上的函数值或进行采样等操作。
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride),
            torch.arange(1, d*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w,d).repeat(1, N, 1, 1,1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w,d).repeat(1, N, 1, 1,1)
        p_0_z = torch.flatten(p_0_z).view(1, 1, h, w,d).repeat(1, N, 1, 1,1)
        p_0 = torch.cat([p_0_x, p_0_y,p_0_z], 1).type(dtype)

        return p_0  # torch.Size([1, 81, 8, 14, 14])

    def _get_p(self, offset, dtype):
        N, h, w,d = offset.size(1)//3, offset.size(2), offset.size(3),offset.size(4)  # torch.Size([2, 81, 8, 14, 14])

        p_n = self._get_p_n(N, dtype).to("cuda")  # 81/3=27
        p_0 = self._get_p_0(h, w,d, N, dtype).to("cuda")
        p = p_0 + p_n + offset
        return p  # torch.Size([2, 81, 8, 14, 14])

    def _get_x_q(self, x, q, N):
        b, h, w,d, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N]*padded_w + q[..., N:2*N]+q[..., 2*N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1,-1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w,d, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w,d, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w,d*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks,d*ks)
        return x_offset

class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.sigmoid(x)


class SConv3D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=8, dropout_rate=0.2):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(SConv3D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool3d, output_size=(1, 1,1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)

      
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv3d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        
        return F.conv3d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _,_ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)

            routing_weights = self._routing_fn(pooled_inputs)

            
            kernels = torch.sum(routing_weights[:, None, None, None, None,None] * self.weight, 0)
            
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)

class Wide_Focus(nn.Module):
    """
    Wide-Focus module.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        self.conv1_1 = nn.Conv3d(in_channels, in_channels, (1, 1, 7), 1, padding=(0, 0, 3))
        self.conv1_2 = nn.Conv3d(in_channels, in_channels, (1, 7, 1), 1, padding=(0, 3, 0))
        self.conv1_3 = nn.Conv3d(in_channels, in_channels, (3, 1, 1), 1, padding=(1, 0, 0))
        self.conv1 = nn.Conv3d(in_channels, out_channels, 1, 1)

        self.conv2_1 = nn.Conv3d(in_channels, in_channels, (1, 1, 5), 1, padding=(0, 0, 2))
        self.conv2_2 = nn.Conv3d(in_channels, in_channels, (1, 5, 1), 1, padding=(0, 2, 0))
        self.conv2_3 = nn.Conv3d(in_channels, in_channels, (5, 1, 1), 1, padding=(2, 0, 0))
        self.conv2 = nn.Conv3d(in_channels, out_channels, 1, 1)

        self.conv3_1 = nn.Conv3d(in_channels, in_channels, (1, 1, 3), 1, padding=(0, 0, 1))
        self.conv3_2 = nn.Conv3d(in_channels, in_channels, (1, 3, 1), 1, padding=(0, 1, 0))
        self.conv3_3 = nn.Conv3d(in_channels, in_channels, (3, 1, 1), 1, padding=(1, 0, 0))
        self.conv3 = nn.Conv3d(in_channels, out_channels, 1, 1)
        
        self.conv1_d = nn.Conv3d(in_channels, out_channels, 3, 1, padding="same", dilation=1)
        self.conv2_d = nn.Conv3d(in_channels, out_channels, 3, 1, padding="same", dilation=2)
        self.conv3_d = nn.Conv3d(in_channels, out_channels, 3, 1, padding="same", dilation=3)

        self.conv4 = nn.Conv3d(out_channels*6, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = F.gelu(x1)
        x1 = self.conv1_2(x1)
        x1 = F.gelu(x1)
        x1 = self.conv1_3(x1)
        x1 = F.gelu(x1)
        # x1 = self.conv1(x1)
        # x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)

        x2 = self.conv2_1(x)
        x2 = F.gelu(x2)
        x2 = self.conv2_2(x2)
        x2 = F.gelu(x2)
        x2 = self.conv2_3(x2)
        x2 = F.gelu(x2)
        # x2 = self.conv2(x2)
        # x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)

        x3 = self.conv3_1(x)
        x3 = F.gelu(x3)
        x3 = self.conv3_2(x3)
        x3 = F.gelu(x3)
        x3 = self.conv3_3(x3)
        x3 = F.gelu(x3)
        # x3 = self.conv3(x3)
        # x3 = F.gelu(x3)
        x3 = F.dropout(x3, 0.1)
        
        
        x4 = self.conv1_d(x)
        x4 = F.gelu(x4)
        x4 = F.dropout(x4, 0.1)
        x5 = self.conv2_d(x)
        x5 = F.gelu(x5)
        x5 = F.dropout(x5, 0.1)
        x6 = self.conv3_d(x)
        x6 = F.gelu(x6)
        x6 = F.dropout(x6, 0.1) 
        # 相加
        # added = torch.add(x1, x2)
        # added = torch.add(added, x3)
        
        # concat
        added = torch.cat((x1,x2,x3,x4,x5,x6), dim=1)
        x_out = self.conv4(added)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, 0.1)
        return x_out



class Wide_Focus2(nn.Module):
    """
    Wide-Focus module.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        self.conv1_d = nn.Conv3d(in_channels, out_channels, 3, 1, padding="same", dilation=1)
        self.conv2_d = nn.Conv3d(in_channels, out_channels, 3, 1, padding="same", dilation=2)
        self.conv3_d = nn.Conv3d(in_channels, out_channels, 3, 1, padding="same", dilation=3)
        self.conv4 = nn.Conv3d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3 = self.conv3(x)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, 0.1)
        added = torch.add(x1, x2)
        added = torch.add(added, x3)
        x_out = self.conv4(added)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, 0.1)
        return x_out


class Block_encoder_bottleneck(nn.Module):
    def __init__(self, blk, in_channels, out_channels, att_heads, dpr, node_num):
        super().__init__()
        self.blk = blk
        if ((self.blk == "first") or (self.blk == "bottleneck")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv3d(in_channels, out_channels*2, 3, 1, padding=1)
            self.conv2 = nn.Conv3d(out_channels*2, out_channels, 3, 1, padding=1)
            self.trans = Transformer(out_channels, att_heads, dpr, node_num)
        elif ((self.blk == "second") or (self.blk == "third") or (self.blk == "fourth")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv2d(1, in_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x, scale_img="none"):
        if ((self.blk == "first") or (self.blk == "bottleneck")):
            x1 = x.permute(0, 2, 3, 4, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 4, 1, 2, 3)
            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))
            x1 = F.dropout(x1, 0.2)
            # x1 = F.max_pool2d(x1, (2, 2))
            out = self.trans(x1)
            # without skip
        elif ((self.blk == "second") or (self.blk == "third") or (self.blk == "fourth")):
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = torch.cat((F.relu(self.conv1(scale_img)), x1), axis=1)
            x1 = F.relu(self.conv2(x1))
            x1 = F.relu(self.conv3(x1))
            x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool2d(x1, (2, 2))
            out = self.trans(x1)
            # with skip
        return out


class Block_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x, skip):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.upsample(x1)
        x1 = F.relu(self.conv1(x1))
        x1 = torch.cat((skip, x1), axis=1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = F.dropout(x1, 0.3)
        out = self.trans(x1)
        return out


class DS_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        out = torch.sigmoid(self.conv3(x1))

        return out


class CPT(nn.Module):
    def __init__(self, depth, height, width, node_num, in_features):
        super().__init__()

        # attention heads and filters per block
        att_heads = [4, 2, 2, 2, 2, 2, 2, 2, 2]
        filters = [320, 16, 32, 64, 128, 64, 32, 16, 8]  # 输出通道

        # number of blocks used in the model
        blocks = len(filters)

        stochastic_depth_rate = 0.0

        # probability for each block
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

        self.drp_out = 0.3

        # shape
        init_sizes = torch.ones((2, 224, 224, 1))
        init_sizes = init_sizes.permute(0, 3, 1, 2)

        # Multi-scale input
        # self.scale_img = nn.AvgPool2d(2, 2)

        # model
        self.block_1 = Block_encoder_bottleneck("first", in_features, in_features, att_heads[0], dpr[0], node_num)


    def forward(self, x):
        x = self.block_1(x)
        out = x
        return out


def init_weights(m):
    """
    Initialize the weights
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # # device = 'cpu'
# depth = 4
# height = 7
# width = 7
# node_num = 196
# in_features = 320
# model = MT(depth, height, width, node_num, in_features).to(device)
# # model.apply(init_weights)
# #
#
# batch_size = 2
# channels = 320
# depth = 4
# length = 7
# wide = 7
# data = torch.randn(batch_size, channels, depth, length, wide)
#
# # print(data.shape)   # torch.Size([1, 1, 320, 196])
#
# data = data.to(device)
# pred = model(data)
# print(pred.shape)
# #



