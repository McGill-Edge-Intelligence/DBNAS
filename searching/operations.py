'''
input shape is : <batch_size, embed_dim, max_len>
operations in the search space, 
search space is inspired by AdaBERT: https://arxiv.org/abs/2001.04246
'''

import torch
import torch.nn as nn
from torch.nn import ReLU
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from itertools import combinations_with_replacement
from data_preprocess import cfg


seq_len = cfg.data.seq_len
embed_dim = cfg.data.embed_dim


# Make sure: the output length is the same as the input length, 2 * p = d * (k - 1), p is pad, k is window_size, d is dilation
OPS = {
    'conv_std_1':lambda embed_dim, seq_len: nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=1, stride=1, padding=get_padding(1, 1), dilation=1),
    'conv_std_3':lambda embed_dim, seq_len: nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=3, stride=1, padding=get_padding(3, 1), dilation=1),
    'conv_std_5':lambda embed_dim, seq_len: nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=5, stride=1, padding=get_padding(5, 1), dilation=1),
    'conv_std_7':lambda embed_dim, seq_len: nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=7, stride=1, padding=get_padding(7, 1), dilation=1),
    'conv_dila_1':lambda embed_dim, seq_len: nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=1, stride=1, padding=get_padding(1, 2), dilation=2),
    'conv_dila_3':lambda embed_dim, seq_len: nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=3, stride=1, padding=get_padding(3, 2), dilation=2),
    'conv_dila_5':lambda embed_dim, seq_len: nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=5, stride=1, padding=get_padding(5, 2), dilation=2),
    'conv_dila_7':lambda embed_dim, seq_len: nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=7, stride=1, padding=get_padding(7, 2), dilation=2),    
    'pool_max_3':lambda embed_dim, seq_len: nn.MaxPool1d(kernel_size=3, stride=1, padding=get_padding(3, 1)),
    'pool_avg_3':lambda embed_dim, seq_len: nn.AvgPool1d(kernel_size=3, stride=1, padding=get_padding(3, 1)),
    'multihead_attention_12':lambda embed_dim, seq_len: nn.MultiheadAttention(embed_dim, num_heads=12),
    'multihead_attention_8':lambda embed_dim, seq_len: nn.MultiheadAttention(embed_dim, num_heads=8),
    'multihead_attention_4':lambda embed_dim, seq_len: nn.MultiheadAttention(embed_dim, num_heads=4)
}


'''
generate a list of random int ranges [0, num_of_ops - 1] of size layers
'''
def uniform_random_op_encoding(num_of_ops, layers=1):
    return np.random.randint(0, num_of_ops, layers)


'''
helper method to calculate the padding the helps retain input shape for conv1d and pooling
'''
def get_padding(kernel_size, dilation):
    return int((kernel_size - 1) * dilation / 2)


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


class one_op_model(pl.LightningModule):
    def __init__(self, model_type):
        super().__init__()
        if 'conv' in model_type:
            self.model = nn.Sequential(
                OPS[model_type](embed_dim, seq_len),
                nn.ReLU(),
                nn.BatchNorm1d(seq_len, affine=True)
            )
        self.model = OPS[model_type](embed_dim, seq_len)
        # self.model = OPS[model_type](embed_dim, seq_len, stride=1, affine=False)

    def forward(self, input):
        return self.model(input)


class MixedOpsDNAS(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.ops_string = list(OPS.keys())
        # print(self.ops_string)

        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(seq_len, affine=True)
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(self.ops_string) for i in range(len(self.ops_string))]),requires_grad=True)
        self.ops = nn.ModuleList([OPS[op_string](embed_dim, seq_len) for op_string in self.ops_string])

    def forward(self,input):
        input = input.type(torch.FloatTensor).to(self.device)
        alphas = nn.functional.gumbel_softmax(self.thetas, tau=0.2)
        outputs = []
        
        for op in self.ops:
            if op.__class__.__name__ == 'MultiheadAttention':
                input_transposed = torch.transpose(input, 0, 1)
                output = torch.transpose(op(input_transposed, input_transposed, input_transposed)[0], 0, 1)
                outputs.append(output)
            # KIM CNN - Conv + ReLU + BN
            elif op.__class__.__name__ == 'Conv1d':
                output = op(input)
                output = self.relu(output)
                output = self.batchnorm(output)
                outputs.append(output)
            else:
                output = op(input)
                outputs.append(output)
        
        output = sum([alpha * output for alpha, output in zip(alphas, outputs)])
        return output