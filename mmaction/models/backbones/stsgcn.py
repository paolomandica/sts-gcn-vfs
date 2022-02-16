# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import math
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..skeleton_gcn.utils import Graph


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 time_dim,
                 joints_dim
                 ):
        super(ConvTemporalGraphical, self).__init__()

        # learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        self.A = nn.Parameter(torch.FloatTensor(
            time_dim, joints_dim, joints_dim))
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(
            joints_dim, time_dim, time_dim))
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)
        '''
        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''

    def forward(self, x):
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        # x=self.prelu(x)
        x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous()


class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True):

        super(ST_GCNN_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        # ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        padding = ((self.kernel_size[0] - 1) // 2,
                   (self.kernel_size[1] - 1) // 2)

        self.gcn = ConvTemporalGraphical(
            time_dim, joints_dim)  # the convolution layer

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride[0], stride[1]),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride[0] != 1 or stride[1] != 1 or in_channels != out_channels:

            self.residual = nn.Sequential(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride[0], stride[1])),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.residual = nn.Identity()

        self.prelu = nn.PReLU()

    def forward(self, x):
     #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x+res
        x = self.prelu(x)
        return x


class CNN_layer(nn.Module):  # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):

        super(CNN_layer, self).__init__()
        self.kernel_size = kernel_size
        # padding so that both dimensions are maintained
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        self.block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(
            out_channels), nn.Dropout(dropout, inplace=True)]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):

        output = self.block(x)
        return output


@BACKBONES.register_module()
class STSGCN(nn.Module):
    """ 
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self,
                 input_channels,
                 input_time_frame,
                 st_gcnn_dropout,
                 joints_to_consider,
                 siamese=False,
                 pretrained=None,
                 freeze=False,
                 #  output_time_frame,
                 #  n_txcnn_layers,
                 #  txc_kernel_size,
                 #  txc_dropout,
                 bias=True):

        super().__init__()
        self.input_time_frame = input_time_frame
        # self.output_time_frame=output_time_frame
        self.joints_to_consider = joints_to_consider
        self.st_gcnns = nn.ModuleList()
        # self.n_txcnn_layers=n_txcnn_layers
        # self.txcnns=nn.ModuleList()

        self.st_gcnns.append(ST_GCNN_layer(input_channels, 32, [1, 1], [1, 1], self.input_time_frame,
                                           joints_to_consider, st_gcnn_dropout))
        self.st_gcnns.append(ST_GCNN_layer(32, 64, [1, 1], [1, 1], self.input_time_frame,
                                           joints_to_consider, st_gcnn_dropout))

        self.st_gcnns.append(ST_GCNN_layer(64, 128, [1, 1], [1, 1], self.input_time_frame,
                                           joints_to_consider, st_gcnn_dropout))

        self.st_gcnns.append(ST_GCNN_layer(128, 256, [1, 1], [1, 1], self.input_time_frame,
                                           joints_to_consider, st_gcnn_dropout))

        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)
        # self.txcnns.append(CNN_layer(input_time_frame,output_time_frame,txc_kernel_size,txc_dropout)) # with kernel_size[3,3] the dimensinons of C,V will be maintained
        # for i in range(1,n_txcnn_layers):
        #     self.txcnns.append(CNN_layer(output_time_frame,output_time_frame,txc_kernel_size,txc_dropout))

        # self.prelus = nn.ModuleList()
        # for j in range(n_txcnn_layers):
        #     self.prelus.append(nn.PReLU())
        self.pretrained = pretrained
        self.freeze = freeze

    def freeze_weights(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.freeze:
            self.requires_grad_(requires_grad=False)
            self.eval()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained,
                            strict=False,
                            logger=logger,
                            revise_keys=[(r'^backbone\.', '')])

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):

        # breakpoint()

        # data normalization
        x = x.float()
        n, c, t, v, m = x.size()  # bs 3 300 25(17) 2
        # x = x.permute(0, 4, 3, 1, 2).contiguous()  # N M V C T
        # x = x.view(n * m, v * c, t)
        # x = self.data_bn(x)
        # x = x.view(n, m, v, c, t)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(n * m, c, t, v)  # bsx2 3 300 25(17)

        for gcn in (self.st_gcnns):
            x = gcn(x)

        # x= x.permute(0,2,1,3) # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)

        # x=self.prelus[0](self.txcnns[0](x))

        # for i in range(1,self.n_txcnn_layers):
        #     x = self.prelus[i](self.txcnns[i](x)) +x # residual connection

        return x
