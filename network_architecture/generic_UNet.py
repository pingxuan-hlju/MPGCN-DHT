#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
from .MHTS import MHTS
from nnunet.network_architecture.Med3D.PMT import ImageProcessor
import torch.nn.functional as F
import torch
class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    用于定义一个卷积层（包含dropout、归一化和非线性激活函数）。
    """

    def __init__(self, input_channels, output_channels,  # 输入和输出通道的数量。
                 conv_op=nn.Conv2d, conv_kwargs=None,  #卷积操作的类，默认为nn.Conv2d。
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,  # 是归一化操作的类，默认为nn.BatchNorm2d。
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,  # 是dropout操作的类
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):  # 是非线性激活函数的类，默认为nn.LeakyReLU。
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        #创建卷积层、dropout层、归一化层和非线性激活函数，并将它们保存到类的成员变量中。
        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):  #它重写了父类的forward方法，对前向传播的顺序进行了微调，
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))   #在非线性激活函数和归一化层的顺序上有所变化。该类首先应用非线性激活函数，然后再进行归一化操作。


class StackedConvLayers(nn.Module):  # 用于定义堆叠的卷积层。
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        #  接下来，根据参数配置构建堆叠的卷积层。首先创建第一个卷积层，并根据first_stride参数设置其步长。然后使用basic_block创建剩余的卷积层，数量为num_convs - 1。
        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,  # 创建第一个卷积层，并根据first_stride参数设置其步长
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    #打印给定模块的训练状态。
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)  #如果module属于上述任何一种类型，则打印该模块的字符串表示和training属性的值


class Upsample(nn.Module):
    #用于执行上采样操作。
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners  #是个布尔值，指定是否对齐角点。默认为False。
        self.mode = mode  # 采样的模式，可以是'nearest'、'linear'、'bilinear'、'bicubic'、'trilinear'之一。默认为'nearest'，使用最近邻插值。
        self.scale_factor = scale_factor  # 尺度因子，可以是一个浮点数或元组。如果为浮点数，则将输入的空间维度缩放为(input_size * scale_factor)。
        #如果为元组，则将输入的空间维度缩放为与scale_factor相匹配。通过尺度因子进行上采样可以在不知道具体目标尺寸的情况下进行操作，只需要提供一个缩放比例即可。
        self.size = size  #目标尺寸，可以是一个整数或元组。如果为整数，则将输入的空间维度调整为(size, size)。如果为元组，则将输入的空间维度调整为与size相匹配。
        #两种方式可以根据具体的需求选择使用。如果已经知道上采样后的目标尺寸，可以使用size参数直接指定。如果只知道需要进行缩放操作，但不确定具体的目标尺寸，可以使用scale_factor参数进行缩放。

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)  #根据上述参数对输入张量进行插值操作，并返回上采样后的结果。


class Generic_UNet(SegmentationNetwork):
    #这段代码定义了一个名为Generic_UNet的类，该类继承自SegmentationNetwork。
    DEFAULT_BATCH_SIZE_3D = 2  #默认的3D批处理大小为2。
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)  # 默认的3D补丁大小为(64, 192, 160)。
    SPACING_FACTOR_BETWEEN_STAGES = 2  # 阶段之间的间距因子为2。
    BASE_NUM_FEATURES_3D = 30  # 3D基础特征数为30。
    MAX_NUMPOOL_3D = 999  # 3D最大池化层数为999。
    MAX_NUM_FILTERS_3D = 320  # 3D最大滤波器数为320。

    DEFAULT_PATCH_SIZE_2D = (256, 256)  # 默认的2D补丁大小为(256, 256)。
    BASE_NUM_FEATURES_2D = 30  # 2D基础特征数为30。
    DEFAULT_BATCH_SIZE_2D = 50  # 默认的2D批处理大小为50。
    MAX_NUMPOOL_2D = 999  # 2D最大池化层数为999。
    MAX_FILTERS_2D = 480  # 2D最大滤波器数为480。

    use_this_for_batch_size_computation_2D = 19739648  # 用于计算2D批处理大小的值为19739648。
    use_this_for_batch_size_computation_3D = 520000000  # 用于计算3D批处理大小的值为520000000。

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        通用的UNet模型的初始化函数。UNet是一种常用的图像分割模型，用于将输入图像分割成多个类别。
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.
        input_channels：输入图像的通道数。
        base_num_features：基础的特征数量，用于控制模型的容量。
        num_classes：要分割的类别数量。
        num_pool：下采样（池化）操作的次数。
        num_conv_per_stage：每个阶段（层）的卷积层数量。
        feat_map_mul_on_downscale：下采样时特征图数量的乘法因子。
        conv_op：卷积操作的类型，可以是2D或3D卷积。
        norm_op：归一化操作的类型。
        norm_op_kwargs：归一化操作的参数。
        dropout_op：随机失活操作的类型。
        dropout_op_kwargs：随机失活操作的参数。
        nonlin：非线性激活函数的类型。
        nonlin_kwargs：非线性激活函数的参数。
        deep_supervision：是否使用深度监督。
        dropout_in_localization：是否在定位路径中使用随机失活。
        final_nonlin：最终输出的非线性激活函数。
        weightInitializer：权重初始化器。
        pool_op_kernel_sizes：池化操作的内核大小。
        conv_kernel_sizes：卷积操作的内核大小。
        upscale_logits：是否对输出进行上采样。
        convolutional_pooling：是否使用卷积池化。
        convolutional_upsampling：是否使用卷积上采样。
        max_num_features：特征数量的最大值。
        basic_block：基本的卷积块类型。
        seg_output_use_bias：分割输出是否使用偏置。
        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:  # 根据conv_op的值选择相应的操作模式（2D或3D卷积）
            upsample_mode = 'bilinear'  # 设置了相应的上采样模式、池化操作、转置卷积等。
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
                #在UNet网络结构中，卷积操作的数量比池化操作的数量多1是因为在网络的上采样路径中，每个池化操作后都会进行一次上采样操作，而这个上采样操作需要一个额外的卷积操
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))
        #通过这段代码，可以获得输入形状必须满足的约束条件。这个条件要求输入形状的高度和宽度都能够被池化核大小的高度和宽度整除，以确保在池化操作中没有截断或溢出。
        #这有助于保持特征图的尺寸和空间信息的一致性，并有助于网络的稳定性和性能。
        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []  #用于存储卷积操作的填充大小。
        for krnl in self.conv_kernel_sizes:  #循环遍历self.conv_kernel_sizes列表中的每个元组，其中每个元组表示一个卷积操作的内核大小。
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])  
            #根据内核大小的元素值来确定填充大小。对于内核大小的元素值为3的情况，填充大小设置为1；否则，填充大小设置为0。这里使用了列表推导式来生成填充大小列表。

        if max_num_features is None:  #即是否没有显式指定最大特征数量。如果是，则根据conv_op的类型设置默认的最大特征数量。
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []  # 用于存储上下文卷积块。这些卷积块通常用于编码器部分，用于提取高级特征和上下文信息。
        self.conv_blocks_localization = []  #用于存储定位卷积块。这些卷积块通常用于解码器部分，用于恢复分辨率和生成分割结果。
        self.td = []  #用于存储下采样操作。这些操作通常是池化操作或其他降采样操作，用于减小特征图的尺寸。
        self.tu = []  #用于存储上采样操作。这些操作通常是反卷积操作或其他上采样操作，用于增加特征图的尺寸。
        self.seg_outputs = []  # 用于存储分割输出。每个元素表示一个分割任务的输出。

        output_features = base_num_features  #output_features是一个变量，表示基础特征的数量。这通常是网络中的初始通道数。
        input_features = input_channels  #是一个变量，表示输入数据的通道数。

        for d in range(num_pool): #用于迭代编码器中的每个下采样阶段。
            # determine the first stride 这段代码用于构建编码器部分的卷积块。
            if d != 0 and self.convolutional_pooling:  
            #在循环中，首先确定第一个步长（stride）的值。如果 d 不为0且 self.convolutional_pooling 为True，表示使用卷积进行池化操作，则第一个步长为上一个池化操作的内核大小
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None
            #设置卷积的参数，包括内核大小（kernel_size）和填充大小（padding）。
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]  
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            #创建一个卷积块对象，并将其添加到self.conv_blocks_context列表中。这个卷积块对象接收一系列参数来配置卷积操作、规范化操作、丢弃操作、激活函数等
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                #如果 self.convolutional_pooling 为False，表示使用非卷积操作进行池化。则将相应的池化操作（如pool_op(pool_op_kernel_sizes[d])）添加到self.td列表中。
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features  #更新 input_features 的值为 output_features，表示下一个阶段的输入特征数。
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            #更新 output_features 的值为 output_features 乘以 feat_map_mul_on_downscale 的四舍五入结果，并转换为整数。

            output_features = min(output_features, self.max_num_features)  
            #将 output_features 的值限制在 self.max_num_features 的范围内，以确保不超过最大特征数量的限制。

        # now the bottleneck.用于确定瓶颈层（bottleneck）的参数设置。瓶颈层在编码器和解码器之间起着连接的作用
        # determine the first stride首先确定瓶颈层的第一个步长（stride）的值。
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]  #第一个步长设为列表 pool_op_kernel_sizes 中最后一个元素（pool_op_kernel_sizes[-1]）
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv接下来，根据是否使用卷积进行上采样操作，确定最终的特征数量。
        if self.convolutional_upsampling:  #示表使用卷积进行上采样。则最终的特征数量设为 output_features，即当前阶段的输出特征数。
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels  #则最终的特征数量设为编码器最后一个卷积块的输出通道数

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]  #设置卷积的内核大小为 self.conv_kernel_sizes 列表中索引为 num_pool 的元素。
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]  #置卷积的填充大小为 self.conv_pad_sizes 列表中索引为 num_pool 的元素。
        self.conv_blocks_context.append(nn.Sequential(
            # 构建编码器部分的卷积块。使用 nn.Sequential 将两个 StackedConvLayers 对象串联起来。第一个 StackedConvLayers 对象接收一系列参数来配置卷积操作、规范化操作、
            #丢弃操作、激活函数等，用于构建编码器部分的卷积块。第二个 StackedConvLayers 对象用于构建解码器部分的卷积块。
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:  #表示解码器部分的卷积块中不使用丢弃操作。则将 self.dropout_op_kwargs['p'] 的值设为0.0，即将丢弃概率设置为0。
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway用于构建解码器部分的定位路径
        for u in range(num_pool):
            '''
            在每次迭代中，首先确定从下采样路径（编码器部分）传递过来的特征数量 nfeatures_from_down，以及从跳跃连接（skip connection）
            传递过来的特征数量 nfeatures_from_skip。
            通过这段代码，可以构建解码器部分的定位路径。定位路径中使用转置卷积（或上采样）进行特征图的恢复，并通过连接操作将跳跃连接的特征图与解码器部分的特征图进行融合。
            这样的结构用于逐渐恢复特征图的分辨率，并生成最终的分割结果。
            其中，nfeatures_from_skip 的值通过 self.conv_blocks_context[-(2 + u)].output_channels 获取，表示从编码器部分的卷积块中获取倒数第2个开始的输出通道数。
            '''
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                #如果 u 不等于 num_pool - 1 且不使用卷积进行上采样，则最终的特征数量设为 self.conv_blocks_context[-(3 + u)].output_channels，
                #即从编码器部分的卷积块中获取倒数第3个开始的输出通道数。
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip  #否则，最终的特征数量设为 nfeatures_from_skip。

            if not self.convolutional_upsampling:
    #如果不使用卷积进行上采样操作，则在 self.tu 列表中添加一个上采样操作（Upsample）对象。该对象使用缩放因子 pool_op_kernel_sizes[-(u + 1)] 进行上采样，并指定上采样的模式。
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
   #如果使用卷积进行上采样操作，则在 self.tu 列表中添加一个转置卷积（transpconv）对象。该对象使用输入特征数量 nfeatures_from_down、
   #输出特征数量 nfeatures_from_skip，以及内核大小和步长为 pool_op_kernel_sizes[-(u + 1)] 的转置卷积
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))
            #设置卷积的参数，包括内核大小（kernel_size）和填充大小（padding）。这些参数根据当前迭代的索引从事先定义好的列表中获取
            #（如self.conv_kernel_sizes和  self.conv_pad_sizes）。
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                #调用StackedConvLayers类的构造函数，创建一个卷积块对象，并将其添加到self.conv_blocks_localization列表中。这个卷积块对象接收一系列参数来配置卷积操作、
                #规范化操作、丢弃操作、激活函数等。
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
                #解码器部分的定位路径中有两个卷积块，第一个卷积块接收 n_features_after_tu_and_concat 个特征作为输入，输出 nfeatures_from_skip 个特征。
                #第二个卷积块接收 nfeatures_from_skip 个特征作为输入，输出 final_num_features 个特征。
            ))

        for ds in range(len(self.conv_blocks_localization)):
            '''
            在循环中，对解码器部分的最后一层卷积块进行处理，生成分割输出。对于每个定位路径（self.conv_blocks_localization 中的每个元素），获取该定位路径的
            最后一个卷积块的输出通道数（self.conv_blocks_localization[ds][-1].output_channels），并使用 conv_op 进行卷积操作，将输出通道数转换为 
            num_classes（分割类别数），生成分割输出。分割输出通过 conv_op 进行卷积操作，内核大小为 1x1，填充为 0，步长为 1，并可以选择是否使用
            偏置项（seg_output_use_bias）。
            '''
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))
         #根据是否进行上采样操作，构建一个与解码器层数相匹配的上采样操作列表 self.upscale_logits_ops。
        self.upscale_logits_ops = []
       
        #上采样的缩放因子由 cum_upsample 计算得到，其中 cum_upsample 是一个累积乘积数组，用于计算每个上采样层的缩放因子。
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]  
        for usl in range(num_pool - 1):
            if self.upscale_logits:  #如果 self.upscale_logits 为 True，则使用 Upsample 对象进行上采样操作。
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:  #则将上采样操作设置为恒等函数，即输入特征图不进行任何改变。
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly  #将各个模块列表转换为 nn.ModuleList 类型，以便能够正确地注册模块。
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)  #解码器部分的定位路径
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)  #编码器部分的卷积块
        self.td = nn.ModuleList(self.td)  #  编码器部分的下采样操作
        self.tu = nn.ModuleList(self.tu)  #解码器部分的上采样操作
        self.seg_outputs = nn.ModuleList(self.seg_outputs)  #分割输出
        if self.upscale_logits:   #上采样操作（如果使用上采样）
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:  #如果定义了权重初始化器 self.weightInitializer，则将其应用于网络的所有模块
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)
            
        num_layers = 3  # 整个模型的层数
        t_layers = 2  # 单层中transformer层数
        num_heads = 8  # transformer头数
        depth = 4
        height = 7
        width = 7
        node_num = 196  # 结点个数
        in_features = 320  # 特征维度
        dropout = 0.1  # dropout率，越大丢弃越多
        edge_num_types = 8  # 构建超边的数量
        self.lgt = MHTS(num_layers, t_layers, num_heads, depth, height, width, node_num, in_features, dropout, edge_num_types)

        num_layers_2 = 3  # 整个模型的层数
        t_layers_2 = 2  # 单层中transformer层数
        num_heads_2 = 8  # transformer头数
        depth_2 = 8
        height_2 = 14
        width_2 = 14
        node_num_2 = 1568  # 结点个数
        in_features_2 = 320  # 特征维度
        dropout_2 = 0.1  # dropout率，越大丢弃越多
        edge_num_types_2 = 8  # 构建边的维度
        self.lgt2 = MHTS(num_layers_2, t_layers_2, num_heads_2, depth_2, height_2, width_2, node_num_2 , in_features_2, dropout_2, edge_num_types_2)
        
        self.covn_change = nn.Conv3d(96, in_features, kernel_size=3, padding=1)
        
        self.covn_sam = nn.Conv3d(in_features*2, in_features, kernel_size=1)
        
        

    def forward(self, x):
        
        #PMT部分代码
# ==========================================================================================
        target_size = (128, 128, 128)
        processor = ImageProcessor(model_type='vit_b_ori', device='cuda', checkpoint=None, img_size=128)
        stretched_tensor, scale_factor = processor.stretch_tensor(x, target_size)
        image_embedding = processor.predict(stretched_tensor)
        de = x.shape[2] // 16
        he = x.shape[3] // 16
        wi = x.shape[4] // 16
        unstretched_tensor = processor.unstretch_tensor(image_embedding, scale_factor,de,he,wi)  #  # torch.Size([2, 384, 2, 14, 14]) 224/16
        
        # unstretched_tensor = torch.nn.functional.interpolate(unstretched_tensor, size=(8,14,14),
        #                                                      mode='trilinear', align_corners=False) #  # torch.Size([2, 384, 8, 14, 14]) 
        bs,channel,depth,height,width = unstretched_tensor.size()
        unstretched_tensor = unstretched_tensor.reshape(bs,96,8,height,width)  # torch.Size([2, 96, 8, 14, 14])  384/4=96 2*4=8
        
        unstretched_tensor =  self.covn_change(unstretched_tensor)  # torch.Size([2, 96, 8, 14, 14]) 
        
# ==========================================================================================
        
        


        # print(unstretched_tensor.shape)  # torch.Size([2, 384, 2, 14, 14])
        
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):  #使用循环遍历编码器部分的卷积块（self.conv_blocks_context）中的所有卷积层（除了最后一层）。
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        # print(x.shape,'x===========')  torch.Size([1, 320, 8, 14, 14]) 卷积之前处理
        
        #   PMT融合代码
# ==========================================================================================
        begin = x
        x = torch.cat((x,unstretched_tensor), dim=1)
        x = self.covn_sam(x)
        x = x + begin
# ==========================================================================================
        
        x_shape = x
        x1 = x.reshape(x.shape[0],x.shape[1],-1)  # 1*320*196
        x = self.lgt2(x1).reshape(x_shape.shape) # 将倒数第二层编码层进模型
        
        skips[-1]= x
        x = self.conv_blocks_context[-1](x)  #将输入 x 通过编码器部分的最后一层卷积块 self.conv_blocks_context[-1] 进行卷积操作，得到输出 x。
        
        x_shape = x
        x1 = x.reshape(x.shape[0],x.shape[1],-1)  # 1*320*196
        x = self.lgt(x1).reshape(x_shape.shape)  # 将最后一层编码层进模型

        for u in range(len(self.tu)):  #使用循环遍历解码器部分的上采样操作 self.tu 列表。
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)  
            #将上采样后的特征图 x 与对应的跳跃连接的特征图 skips[-(u + 1)] 进行通道拼接操作（torch.cat），拼接维度为 1（通道维度）。
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            #如果启用了深度监督（self._deep_supervision 为 True）且需要进行深度监督（self.do_ds 为 True），则返回多个分割输出。
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        用于估计网络在计算机显存（VRAM）中的近似消耗。它计算了网络的参数存储量和特征图的近似大小，并返回一个与显存消耗大致成比例的常数项。
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):  #检查 num_pool_per_axis 是否为 np.ndarray 类型，如果不是，则将其转换为 np.ndarray 类型。
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)  #初始化变量 npool，表示池化操作的数量

        map_size = np.array(patch_size)  #将 patch_size 转换为 np.array 类型
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))
        #计算网络的初始显存消耗，存储在变量 tmp 中：
        #计算卷积层的显存消耗，包括编码器和解码器部分的卷积层以及转置卷积层。根据每个阶段的卷积数 conv_per_stage，将映射大小 map_size 乘以基础特征数量 base_num_features。
        #计算输入的显存消耗，根据输入的通道数 num_modalities 和映射大小 map_size 计算得到。
        #计算输出的显存消耗，根据输出的类别数 num_classes 和映射大小 map_size 计算得到。

        num_feat = base_num_features  #初始化变量 num_feat，表示当前的特征数量，初始值为基础特征数量 base_num_features。

        for p in range(npool):  #对于每个池化操作 p，更新映射大小 map_size 和特征数量 num_feat：
            for pi in range(len(num_pool_per_axis)):  #将映射大小 map_size 按照 pool_op_kernel_sizes 进行相应轴的缩小。
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)  #更新特征数量 num_feat，将其乘以 2，但不超过最大特征数量 max_num_features。
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            #更新显存消耗 tmp，将其增加卷积块数量 num_blocks 与映射大小 map_size 和特征数量 num_feat 的乘积。
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat  
            if deep_supervision and p < (npool - 2):
                #如果启用了深度监督（deep_supervision 为 True）且当前池化操作 p 不是倒数第二个池化操作，则增加一个用于深度监督的显存消耗，计算方式与输出的显存消耗类似。
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
