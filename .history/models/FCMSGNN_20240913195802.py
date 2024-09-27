import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers.FCMSGBlock import *


#### Best for HAR
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.conv_out = configs.conv_out
        self.lstmhidden_dim = configs.lstmhidden_dim
        self.lstmout_dim = configs.lstmout_dim
        self.conv_kernel = configs.conv_kernel
        self.hidden_dim = configs.hidden_dim
        self.time_length = configs.time_denpen_len
        self.num_nodes = configs.num_nodes
        self.num_windows = configs.num_windows
        self.moving_window = configs.moving_window
        self.stride = configs.stride
        self.decay = configs.decay
        self.pooling_choice = configs.pooling_choice
        self.n_class = configs.n_class
        
        # graph_construction_type = args.graph_construction_type        
        # 非线性映射模块，用于特征提取
        self.nonlin_map = Feature_extractor_1DCNN_HAR_SSC(1, self.lstmhidden_dim, self.lstmout_dim, kernel_size=self.conv_kernel)
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(self.lstmout_dim*self.conv_out, 2*self.hidden_dim),
            nn.BatchNorm1d(2*self.hidden_dim))   # 180 → 32
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(2*self.hidden_dim, 0.1, max_len=5000)
        
        # 图构建和聚合：图卷积池化MPNN模块
        self.MPNN1 = GraphConvpoolMPNN_block_v6(2*self.hidden_dim, self.hidden_dim, self.num_nodes, self.time_length, 
                                                time_window_size=self.moving_window[0], stride=self.stride[0], 
                                                decay=self.decay, pool_choice=self.pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(2*self.hidden_dim, self.hidden_dim, self.num_nodes, self.time_length, 
                                                time_window_size=self.moving_window[1], stride=self.stride[1], 
                                                decay=self.decay, pool_choice=self.pooling_choice)       
        # FC Graph Convolution
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.hidden_dim * self.num_windows * self.num_nodes, 2*self.hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2*self.hidden_dim, self.hidden_dim)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc4', nn.Linear(self.hidden_dim, self.n_class)),
        ]))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    # def forward(self, X):
        bs, dimension, num_nodes = x_enc.size()                        # 100, 2, 9, 64   # 32, 96, 7

        ### Graph Generation
        A_input = torch.reshape(x_enc, [bs*num_nodes, dimension, 1])    # [1800, 64, 1]  # [224, 96, 1]
        A_input_ = self.nonlin_map(A_input)                             # [1800, 18, 10] # [224, 18, 14]
        A_input_ = torch.reshape(A_input_, [bs*num_nodes, -1])          # [1800, 180]    # [224, 252]
        A_input_ = self.nonlin_map2(A_input_)                           # [1800, 32]
        A_input_ = torch.reshape(A_input_, [bs, num_nodes, -1])         # [100, 2, 9, 32]

        ## positional encoding before mapping starting
        X_ = torch.reshape(A_input_, [bs, num_nodes, -1])          # [100, 2, 9, 32]
        X_ = torch.transpose(X_, 1, 2)
        X_ = torch.reshape(X_, [bs*num_nodes, -1])                 # [900, 2, 32]
        X_ = self.positional_encoding(X_)
        X_ = torch.reshape(X_, [bs, num_nodes, -1])
        X_ = torch.transpose(X_, 1, 2)
        A_input_ = X_                                                    # [100, 2, 9, 32]

        ## positional encoding before mapping ending
        MPNN_output1 = self.MPNN1(A_input_)                     # [100, 1, 9, 16]
        MPNN_output2 = self.MPNN2(A_input_)                     # [100, 1, 9, 16]

        features1 = torch.reshape(MPNN_output1, [bs, -1])       # [100, 144]
        features2 = torch.reshape(MPNN_output2, [bs, -1])       # [100, 144]
        features = torch.cat([features1,features2], -1)         # [100, 288]
        
        features = self.fc(features)                            # [100, 6]
        return features