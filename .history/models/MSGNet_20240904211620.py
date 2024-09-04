import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import pywt
from layers.Embed import DataEmbedding
from layers.MSGBlock import GraphBlock, simpleVIT, Attention_Block, Predict


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
        scale_list, scale_weight = FFT_for_Period(x, self.k)    # scale_weight：[32, 5]
        res = []
        for i in range(self.k):
            scale = scale_list[i]
            
            #Gconv
            x = self.gconv[i](x)           
            # padding
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x     # [32, 96, 512]
            out = out.reshape(B, length // scale, scale, N) # [32, 1, 96, 512]

            #for Mul-attetion
            out = out.reshape(-1 , scale , N)   # [32, 96, 512]
            out = self.norm(self.att0(out))     # [32, 96, 512]
            out = self.gelu(out)
            out = out.reshape(B, -1 , scale , N).reshape(B ,-1 ,N)
            
        #   #for simpleVIT
        #     out = self.att(out.permute(0, 3, 1, 2).contiguous()) #return
        #     out = out.permute(0, 2, 3, 1).reshape(B, -1 ,N)

            out = out[:, :self.seq_len, :]
            res.append(out)
        res = torch.stack(res, dim=-1)
        
        # adaptive aggregation
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        
        # residual connection
        res = res + x       
        return res


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # for graph
        # self.num_nodes = configs.c_out
        # self.subgraph_size = configs.subgraph_size
        # self.node_dim = configs.node_dim
        # to return adj (node , node)
        # self.graph = constructor_graph()

        self.model = nn.ModuleList([ScaleGraphBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.seq2pred = Predict(configs.individual ,configs.c_out,configs.seq_len, configs.pred_len, configs.dropout)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc:[32, 96, 7],x_mark_enc:[32, 96, 4],x_dec:[32, 144, 7],x_mark_dec:[32, 144, 4]
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)         # [32, 96, 512]
        # adp = self.graph(torch.arange(self.num_nodes).to(self.device))
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))   # [32, 96, 512]

        # porject back
        dec_out = self.projection(enc_out)
        dec_out = self.seq2pred(dec_out.transpose(1, 2)).transpose(1, 2)    # [32, 96, 7]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out[:, -self.pred_len:, :]   # [32, 96, 7]