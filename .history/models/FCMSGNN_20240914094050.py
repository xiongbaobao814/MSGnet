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
        bs, dimension, num_nodes = x_enc.size()                        # 100, 2, 9, 64      # 32, 96, 7

        ### Graph Generation
        A_input = torch.reshape(x_enc, [bs*num_nodes, dimension, 1])    # [1800, 64, 1]     # [224, 96, 1]
        A_input_ = self.nonlin_map(A_input)                             # [1800, 18, 10]    # [224, 18, 14]
        A_input_ = torch.reshape(A_input_, [bs*num_nodes, -1])          # [1800, 180]       # [224, 252]
        A_input_ = self.nonlin_map2(A_input_)                           # [1800, 32]        # [224, 32]
        A_input_ = torch.reshape(A_input_, [bs, num_nodes, -1])         # [100, 2, 9, 32]   # [32, 7, 32]

        ## positional encoding before mapping starting
        X_ = torch.reshape(A_input_, [bs, num_nodes, -1])          # [100, 2, 9, 32]        # [32, 7, 32]
        # X_ = torch.transpose(X_, 1, 2)
        # X_ = torch.reshape(X_, [bs*num_nodes, -1])                 # [900, 2, 32]         
        X_ = self.positional_encoding(X_)
        X_ = torch.reshape(X_, [bs, num_nodes, -1])                                           # [32, 7, 32]
        # X_ = torch.transpose(X_, 1, 2)
        A_input_ = X_                                              # [100, 2, 9, 32]          # [32, 7, 32]

        ## Graph Convolution
        MPNN_output1 = self.MPNN1(A_input_)                     # [100, 1, 9, 16]
        MPNN_output2 = self.MPNN2(A_input_)                     # [100, 1, 9, 16]

        features1 = torch.reshape(MPNN_output1, [bs, -1])       # [100, 144]
        features2 = torch.reshape(MPNN_output2, [bs, -1])       # [100, 144]
        features = torch.cat([features1,features2], -1)         # [100, 288]
        
        features = self.fc(features)                            # [100, 6]
        return features
    

def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)                   # [32, 49, 512],对输入x沿维度1进行FFT变换得到频域表示
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)     # 找出前k个最高频率分量的索引
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list                 # 根据这些索引计算信号可能的周期长度period,[96, 48, 32, 24, 19]
    return period, abs(xf).mean(-1)[:, top_list]    # 对应于top_list中频率分量的平均幅度值


class ScaleGraphBlock(nn.Module):
    def __init__(self, configs):
        super(ScaleGraphBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.att0 = Attention_Block(configs.d_model, configs.d_ff, n_heads=configs.n_heads, 
                                    dropout=configs.dropout, activation="gelu")
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList()
        for i in range(self.k):
            self.gconv.append(GraphBlock(configs.c_out, configs.d_model, configs.conv_channel, 
                                        configs.skip_channel, configs.gcn_depth, configs.dropout, 
                                        configs.propalpha ,configs.seq_len, configs.node_dim))

    def forward(self, x):
        B, T, N = x.size()  # [32, 96, 512]
        scale_list, scale_weight = FFT_for_Period(x, self.k)    # scale_weight:[32, 5]
        res = []
        for i in range(self.k):
            scale = scale_list[i]                               # scale_list:[96, 48, 32, 24, 19]        
            # Gconv
            x = self.gconv[i](x)                                # [32, 96, 512]

            # padding
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x     # [32, 96, 512]
            out = out.reshape(B, length // scale, scale, N) # [32, 1, 96, 512]

            # Multi-attetion
            out = out.reshape(-1 , scale , N)   # [32, 96, 512]
            out = self.norm(self.att0(out))     # [32, 96, 512]
            out = self.gelu(out)
            out = out.reshape(B, -1 , scale , N).reshape(B ,-1 ,N)  # [32, 96, 512]
            
        #   #for simpleVIT
        #     out = self.att(out.permute(0, 3, 1, 2).contiguous()) #return
        #     out = out.permute(0, 2, 3, 1).reshape(B, -1 ,N)

            out = out[:, :self.seq_len, :]      # [32, 96, 512]
            res.append(out)
        res = torch.stack(res, dim=-1)
        
        # adaptive aggregation
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        
        # residual connection
        res = res + x       
        return res


# 加入patch变换后    
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
        self.k = configs.top_k
        
        # graph_construction_type = args.graph_construction_type        
        # 非线性映射模块，用于特征提取
        self.nonlin_map = Feature_extractor_1DCNN_HAR_SSC(1, self.lstmhidden_dim, self.lstmout_dim, kernel_size=self.conv_kernel)
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(self.lstmout_dim*self.conv_out, 2*self.hidden_dim),
            nn.BatchNorm1d(2*self.hidden_dim))   # 180 → 32
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(2*self.hidden_dim, 0.1, max_len=5000)
        
        self.scalegraph = ScaleGraphBlock(configs)
        
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
        bs, dimension, num_nodes = x_enc.size()                        # 100, 2, 9, 64      # 32, 96, 7

        ### Graph Generation
        A_input = torch.reshape(x_enc, [bs*num_nodes, dimension, 1])    # [1800, 64, 1]     # [224, 96, 1]
        A_input_ = self.nonlin_map(A_input)                             # [1800, 18, 10]    # [224, 18, 14]
        A_input_ = torch.reshape(A_input_, [bs*num_nodes, -1])          # [1800, 180]       # [224, 252]
        A_input_ = self.nonlin_map2(A_input_)                           # [1800, 32]        # [224, 32]
        A_input_ = torch.reshape(A_input_, [bs, num_nodes, -1])         # [100, 2, 9, 32]   # [32, 7, 32]

        ## positional encoding
        X_ = torch.reshape(A_input_, [bs, num_nodes, -1])          # [100, 2, 9, 32]        # [32, 7, 32]
        # X_ = torch.transpose(X_, 1, 2)
        # X_ = torch.reshape(X_, [bs*num_nodes, -1])                 # [900, 2, 32]         
        X_ = self.positional_encoding(X_)
        X_ = torch.reshape(X_, [bs, num_nodes, -1])                                           # [32, 7, 32]
        # X_ = torch.transpose(X_, 1, 2)
        A_input_ = X_                                              # [100, 2, 9, 32]          # [32, 7, 32]

        ## Graph Convolution
        
        self.scalegraph
        MPNN_output1 = self.MPNN1(A_input_)                     # [100, 1, 9, 16]
        MPNN_output2 = self.MPNN2(A_input_)                     # [100, 1, 9, 16]

        features1 = torch.reshape(MPNN_output1, [bs, -1])       # [100, 144]
        features2 = torch.reshape(MPNN_output2, [bs, -1])       # [100, 144]
        features = torch.cat([features1,features2], -1)         # [100, 288]
        
        features = self.fc(features)                            # [100, 6]
        return features