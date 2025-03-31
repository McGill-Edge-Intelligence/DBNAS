'''
input shape is : <batch_size, embed_dim, max_len>
operations in the search space, 
search space is inspired by AdaBERT: https://arxiv.org/abs/2001.04246
'''

import torch
import torch.nn as nn
from torch.nn import ReLU
import torch.nn.functional as F
import pytorch_lightning as pl
from itertools import combinations_with_replacement
from data_preprocess import cfg

import numpy as np
import math


seq_len = cfg.data.seq_len
embed_dim = cfg.data.embed_dim


# Make sure: the output length is the same as the input length, 2 * p = d * (k - 1), p is pad, k is window_size, d is dilation
OPS = {
    'conv_std_1': lambda embed_dim, seq_len, stride, affine: \
        TextConv(in_channels=seq_len, out_channels=seq_len, kernel_size=1, stride=1, dilation=1, affine=affine),
    'conv_std_3': lambda embed_dim, seq_len, stride, affine: \
        TextConv(in_channels=seq_len, out_channels=seq_len, kernel_size=3, stride=1, dilation=1, affine=affine),
    'conv_std_5': lambda embed_dim, seq_len, stride, affine: \
        TextConv(in_channels=seq_len, out_channels=seq_len, kernel_size=5, stride=1, dilation=1, affine=affine),
    'conv_std_7': lambda embed_dim, seq_len, stride, affine: \
        TextConv(in_channels=seq_len, out_channels=seq_len, kernel_size=7, stride=1, dilation=1, affine=affine),
    'conv_dila_1': lambda embed_dim, seq_len, stride, affine: \
        TextConv(in_channels=seq_len, out_channels=seq_len, kernel_size=1, stride=1, dilation=2, affine=affine),
    'conv_dila_3': lambda embed_dim, seq_len, stride, affine: \
        TextConv(in_channels=seq_len, out_channels=seq_len, kernel_size=3, stride=1, dilation=2, affine=affine),
    'conv_dila_5': lambda embed_dim, seq_len, stride, affine: \
        TextConv(in_channels=seq_len, out_channels=seq_len, kernel_size=5, stride=1, dilation=2, affine=affine),
    'conv_dila_7': lambda embed_dim, seq_len, stride, affine: \
        TextConv(in_channels=seq_len, out_channels=seq_len, kernel_size=7, stride=1, dilation=2, affine=affine),
    'pooling_ave_3': lambda embed_dim, seq_len, stride, affine: \
        PoolBN(pool_type='avg', kernel_size=3, channels=seq_len, stride=1, padding=1, affine=affine),
    'pooling_max_3': lambda embed_dim, seq_len, stride, affine: \
        PoolBN(pool_type='max', kernel_size=3, channels=seq_len, stride=1, padding=1, affine=affine),
    'multihead_attention': lambda embed_dim, seq_len, stride, affine: nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8)
}

ops_dict = {
    0: 'conv',
    1: 'pooling',
    2: 'multihead_attention'
}

conv_dict = {
    0: 'conv_std_1',
    1: 'conv_std_3',
    2: 'conv_std_5',
    3: 'conv_std_7',
    4: 'conv_dila_1',
    5: 'conv_dila_3',
    6: 'conv_dila_5',
    7: 'conv_dila_7'
}

pool_dict = {
    0: 'pooling_ave_3',
    1: 'pooling_max_3'
}

'''
generate a list of random int ranges [0,num_of_ops-1] of size layers
'''
def uniform_random_op_encoding(num_of_ops, layers=1):
    return np.random.randint(0, num_of_ops, layers)


'''
helper method to calculate the padding the helps retain input shape for conv1d and pooling
'''
def get_padding(kernel_size):
    return int((kernel_size-1)/2)


class TextConv(nn.Module):
    '''
    KIM CNN: Conv - ReLU - BN
    '''
    # kernel_size n indicates n-gram window size
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, affine=True):
        super().__init__()
        # pad to make the input length as the same as the output length
        pad = int((kernel_size -1) * dilation / 2)
        assert 2 * pad == dilation * (kernel_size - 1)  # (dilation is odd and kernel_size is even) is forbidden
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=pad),
            ReLU(),
            nn.BatchNorm1d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class PoolBN(nn.Module):
    '''
    AvgPool or MaxPool - BN
    '''
    def __init__(self, pool_type, kernel_size, channels, stride, padding, affine=True):
        '''
        Args:
            pool_type: select between 'avg or 'max'
        '''
        super().__init__()
        if pool_type.lower() == 'avg':
            self.pool = nn.AvgPool1d(kernel_size, stride, padding, count_include_pad=False)
        elif pool_type.lower() == 'max':
            self.pool = nn.MaxPool1d(kernel_size, stride, padding)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm1d(channels, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0