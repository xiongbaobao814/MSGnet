import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).cuda()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # [224, 32]
        # x = x + torch.Tensor(self.pe[:, :x.size(1)], requires_grad=False)
        # print(self.pe[0, :x.size(1),2:5])
        pes = self.pe                   # [1, 5000, 32]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        # return x


# 定义一个用于提取特征的一维卷积神经网络模型，通过多层卷积和池化操作逐步提取并浓缩输入信号中的特征信息
class Feature_extractor_1DCNN_HAR_SSC(nn.Module):
    def __init__(self, input_channels, num_hidden, embedding_dimension, kernel_size=3, stride=1, dropout=0):
        super(Feature_extractor_1DCNN_HAR_SSC, self).__init__()
        # input_channels → num_hidden: 1 → 48
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size, 
                        stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )
        # num_hidden → num_hidden*2: 48 → 96
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden*2, kernel_size=kernel_size, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(num_hidden*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        # num_hidden*2 → embedding_dimension: 96 → 18
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden*2, embedding_dimension, kernel_size=kernel_size, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(embedding_dimension),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        x = torch.transpose(x_in, -1, -2)            # [224, 1, 96]  [448, 1, 48]
        x = self.conv_block1(x)                      # [224, 48, 49] [448, 48, 25]
        x = self.conv_block2(x)                      #               [448, 96, 13]
        x = self.conv_block3(x)   # [1800, 18, 10]   # [224, 18, 14] [448, 18, 8]
        # print(x.size())
        return x


def Dot_Graph_Construction(node_features):
    ## node features size is (bs, N, dimension)
    ## output size is (bs, N, N)
    bs, N, dimen = node_features.size()

    node_features_1 = torch.transpose(node_features, 1, 2)

    Adj = torch.bmm(node_features, node_features_1)

    eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()
    eyes_like_inf = eyes_like*1e8
    Adj = F.leaky_relu(Adj-eyes_like_inf)
    Adj = F.softmax(Adj, dim = -1)
    # print(Adj[0])
    Adj = Adj+eyes_like
    # print(Adj[0])
    # if prior:
    return Adj


# 构建节点特征间的加权邻接矩阵Adjacent Matrix (边E)
class Dot_Graph_Construction_weights(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):                           # [32, 14, 32]
        # node_features: 表示一批图中的节点特征
        node_features = self.mapping(node_features)             # [100, 18, 32] [32, 14, 32]
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.size()                     # [100, 18, 32]
        # 计算节点特征之间的点积，得到邻接矩阵
        node_features_1 = torch.transpose(node_features, 1, 2)  # [100, 32, 18] [32, 32, 14]
        Adj = torch.bmm(node_features, node_features_1)         # [100, 18, 18] [32, 14, 14]
        # 通过减去一个大数值对角矩阵并应用LeakyReLU激活确保邻接矩阵非对角元素为正，对角元素较小
        eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()
        eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj - eyes_like_inf)
        
        Adj = F.softmax(Adj, dim=-1)
        Adj = Adj + eyes_like
        return Adj


class Dot_Graph_Construction_weights_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, hidden_dim)

    def forward(self, node_features):
        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = torch.transpose(node_features, 1, 2)
        Adj = torch.bmm(node_features, node_features_1)

        eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()
        eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj - eyes_like_inf)
        
        Adj = F.softmax(Adj, dim=-1)
        Adj = Adj + eyes_like
        return Adj


def iDot_Graph_Construction(node_features):
    ## node features size is (bs, N, dimension)
    ## output size is (bs, N, N)
    bs, N, dimen = node_features.size()

    node_features_1 = torch.transpose(node_features, 1, 2)

    Adj = torch.bmm(node_features, node_features_1)

    eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()
    eyes_like_inf = eyes_like*1e8
    Adj = F.leaky_relu(Adj-eyes_like_inf)
    Adj = F.softmax(Adj, dim = -1)
    Adj = Adj+eyes_like

    return Adj


class MPNN_mk(nn.Module):
    def __init__(self, input_dimension, output_dinmension, k):
        ### In GCN, k means the size of receptive field. Different receptive fields can be concatnated or summed
        ### k=1 means the traditional GCN
        super(MPNN_mk, self).__init__()
        self.way_multi_field = 'sum' ## two choices 'cat' (concatnate) or 'sum' (sum up)
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, output_dinmension))
        self.theta = nn.ModuleList(theta)

    def forward(self, X, A):
        ## size of X is (bs, N, A), size of A is (bs, N, N)
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = torch.bmm(A_,A)
            out_k = self.theta[kk](torch.bmm(A_,X))
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            GCN_output_ = torch.cat(GCN_output_, -1)
        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)

        return F.leaky_relu(GCN_output_)


# 在图结构上执行信息传递和更新操作,支持不同大小的感受野k,并可以灵活选择如何组合这些不同大小的感受野
class MPNN_mk_v2(nn.Module):
    def __init__(self, input_dimension, output_dinmension, k):
        ### k表示不同的邻居组合方式，即“感受野”的大小。k控制了邻接矩阵A的累积次数，从而决定了每个节点可以考虑多少层邻居的信息。
        ### k=1 means the traditional GCN
        super(MPNN_mk_v2, self).__init__()
        self.way_multi_field = 'sum'    # 决定多个感受野的结果是拼接还是相加,'cat'(concatnate) or 'sum'(sum up) 
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, output_dinmension))
        self.theta = nn.ModuleList(theta)
        self.bn1 = nn.BatchNorm1d(output_dinmension)

    def forward(self, X, A):    # [32, 14, 32],[32, 14, 14]
        # X:[bs, N, input_dimension],每个节点的特征; A:[bs, N, N],邻接矩阵,节点之间的连接关系
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A                                      #               [32, 14, 14]
            else:
                A_ = torch.bmm(A_, A)   # 通过多次矩阵乘法累积邻居信息
            out_k = self.theta[kk](torch.bmm(A_, X))        # [100, 18, 16] [32, 14, 16]
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            GCN_output_ = torch.cat(GCN_output_, -1)
        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)                  # [100, 18, 16] [32, 14, 16]
        # 批量归一化
        GCN_output_ = torch.transpose(GCN_output_, -1, -2)
        GCN_output_ = self.bn1(GCN_output_)
        GCN_output_ = torch.transpose(GCN_output_, -1, -2)  # [100, 18, 16] [32, 14, 16]

        return F.leaky_relu(GCN_output_)


def Graph_regularization_loss(X, Adj, gamma):
    ### X size is (bs, N, dimension)
    ### Adj size is (bs, N, N)
    X_0 = X.unsqueeze(-3)
    X_1 = X.unsqueeze(-2)

    X_distance = torch.sum((X_0 - X_1)**2, -1)

    Loss_GL_0 = X_distance*Adj
    Loss_GL_0 = torch.mean(Loss_GL_0)
    Loss_GL_1 = torch.sqrt(torch.mean(Adj**2))
    # print('Loss GL 0 is {}'.format(Loss_GL_0))
    # print('Loss GL 1 is {}'.format(Loss_GL_1))
    
    Loss_GL = Loss_GL_0 + gamma*Loss_GL_1
    return Loss_GL


# 将原始时间序列转换成一系列特定大小的时间滑动窗口
def Conv_GraphST(input, moving_window, stride):     # [32, 2, 7, 32]
    
    bs, scale_num, num_nodes, feature_dim = input.size()                     # [100, 2, 9, 32]  [32, 2, 7, 32]
    x_ = torch.transpose(input, 1, 3)                                        # [100, 32, 9, 2]  [32, 32, 7, 2]
    # 将时间序列转换成一系列重叠或不重叠的子序列
    y_ = F.unfold(x_, (num_nodes, moving_window), stride=stride)             # [100, 576, 1]    [32, 448, 1]
    y_ = torch.reshape(y_, [bs, feature_dim, num_nodes, moving_window, -1])  # [100, 32, 9, 2, 1]   [32, 32, 7, 2, 1]
    y_ = torch.transpose(y_, 1, -1)                                          # [100, 1, 9, 2, 32]   [32, 1, 7, 2, 32]
    return y_


def Conv_GraphST_pad(input, moving_window, stride, padding):
    ## input size is (bs, scale_num, num_sensors, feature_dim)
    ## output size is (bs, num_windows, num_sensors, moving_window, feature_dim)
    bs, scale_num, num_sensors, feature_dim = input.size()
    x_ = torch.transpose(input, 1, 3)

    y_ = F.unfold(x_, (num_sensors, moving_window), stride=stride, padding=[0,padding])
    y_ = torch.reshape(y_, [bs, feature_dim, num_sensors, moving_window, -1])
    y_ = torch.transpose(y_, 1,-1)

    return y_


# 创建衰减矩阵Decay Matrix
def Mask_Matrix(num_node, scale_num, decay_rate):
    # Adj:全1方阵
    Adj = torch.ones(num_node * scale_num, num_node * scale_num).cuda()
    # 使用两层循环遍历所有可能的时间点组合
    for i in range(scale_num):                # 遍历从0到scale_num-1的时间点
        v = 0
        for r_i in range(i, scale_num):       # 1.处理当前时间点与未来时间点的关系
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node    ## 获取当前时间点到未来时间点的索引范围，行范围 
            idx_s_col = (r_i) * num_node
            idx_e_col = (r_i + 1) * num_node  ## 获取当前时间点到未来时间点的索引范围，列范围
            Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (decay_rate ** (v))
            v = v+1
        v=0
        for r_i in range(i+1):                  # 2.处理当前时间点与过去时间点的关系
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node      ## 获取当前时间点到未来时间点的索引范围，行范围
            idx_s_col = (i-r_i) * num_node
            idx_e_col = (i-r_i + 1) * num_node  ## 获取当前时间点到未来时间点的索引范围，列范围
            Adj[idx_s_row:idx_e_row,idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row,idx_s_col:idx_e_col] * (decay_rate ** (v))
            v = v+1
    return Adj


class GraphConvpoolMPNN_block_v6(nn.Module):
    def __init__(self, input_dim, output_dim, num_nodes, scale_num, moving_window, stride, decay, pool_choice):
        super(GraphConvpoolMPNN_block_v6, self).__init__()
        self.moving_window = moving_window
        self.stride = stride
        self.output_dim = output_dim
        # 构造图邻接矩阵
        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        self.BN = nn.BatchNorm1d(input_dim)
        # 执行消息传递和图卷积操作
        self.MPNN = MPNN_mk_v2(input_dim, output_dim, k=1)
        
        # 计算衰减矩阵
        self.pre_relation = Mask_Matrix(num_nodes, moving_window, decay)
        self.pool_choice = pool_choice

    def forward(self, input):   # [32, 1, 96, 512]  [32, 2, 7, 32]
        ## input size:[bs, scale_num, num_nodes, input_dim]
        ## output size:[bs, output_node_t, output_node_s, output_dim]

        # 卷积图处理
        input_con = Conv_GraphST(input, self.moving_window, self.stride)

        bs, num_windows, num_nodes, moving_window, feature_dim = input_con.size()   # [32, 1, 7, 2, 32]
        input_con_ = torch.transpose(input_con, 2, 3)                               # [32, 1, 2, 7, 32]
        input_con_ = torch.reshape(input_con_, [bs*num_windows, moving_window*num_nodes, feature_dim]) # [100, 18, 32]  [32, 14, 32]

        # 构造衰减前的图邻接矩阵与衰减后的图邻接矩阵
        A_input = self.graph_construction(input_con_)       # [100, 18, 18] [32, 14, 14]
        A_input = A_input*self.pre_relation                 #               [32, 14, 14]
        input_con_ = torch.transpose(input_con_, -1, -2)    # [100, 32, 18] [32, 32, 14]
        input_con_ = self.BN(input_con_)
        
        # 图卷积操作
        input_con_ = torch.transpose(input_con_, -1, -2)    # [100, 18, 16] [32, 14, 32]
        X_output = self.MPNN(input_con_, A_input)           #               [32, 14, 16]
        X_output = torch.reshape(X_output, [bs, num_windows, moving_window, num_nodes, self.output_dim])  
        # [100, 1, 2, 9, 16] [32, 1, 2, 7, 16]
        # 图池化操作
        if self.pool_choice == 'mean':
            X_output = torch.mean(X_output, 2)  # [100, 1, 9, 16]
        elif self.pool_choice == 'max':
            X_output, ind = torch.max(X_output, 2)
        else:
            print('input choice for pooling cannot be read')
        # X_output = torch.reshape(X_output, [bs, num_windows*moving_window,num_sensors, self.output_dim])

        return X_output


class MPNN_block_seperate(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, time_conv, moving_window, stride, decay, pool_choice):
        super(MPNN_block_seperate, self).__init__()
        self.moving_window = moving_window
        self.stride = stride
        self.output_dim = output_dim

        self.graph_construction = Dot_Graph_Construction_weights(input_dim*2)
        self.BN = nn.BatchNorm1d(input_dim)

        self.Temporal = Feature_extractor_1DCNN_RUL(input_dim, input_dim*2, input_dim*2,kernel_size=3)
        self.time_conv = time_conv
        self.Spatial = MPNN_mk_v2(2*input_dim, output_dim, k=1)

        self.pre_relation = Mask_Matrix(num_sensors,moving_window,decay)

        self.pool_choice = pool_choice
    def forward(self, input):
        ## input size (bs, scale_num, num_nodes, input_dim)
        bs, scale_num, num_nodes, input_dim = input.size()

        # input_con = Conv_GraphST(input, self.moving_window, self.stride)
        # ## input_con size (bs, num_windows, num_sensors, moving_window, feature_dim)
        # bs, num_windows, num_sensors, moving_window, feature_dim = input_con.size()
        # input_con_ = torch.transpose(input_con, 2,3)
        # input_con_ = torch.reshape(input_con_, [bs*num_windows, moving_window*num_sensors, feature_dim])

        tem_input = torch.transpose(input, 1,2)
        tem_input = torch.reshape(tem_input, [bs*num_nodes, scale_num, input_dim])

        tem_output = self.Temporal(tem_input)
        # print(tem_output.size())
        tem_output = torch.reshape(tem_output, [bs, num_nodes, self.time_conv, 2*input_dim])
        spa_input = torch.transpose(tem_output, 1,2)

        spa_input = torch.reshape(spa_input, [bs*self.time_conv, num_nodes, 2*input_dim])
        A_input = self.graph_construction(spa_input)

        spa_output = self.Spatial(spa_input, A_input)
        return spa_output


class GraphMPNNConv_block(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, moving_window, stride, decay):
        super(GraphMPNNConv_block, self).__init__()
        self.moving_window = moving_window
        self.stride = stride
        self.output_dim = output_dim

        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        self.MPNN = MPNN_mk(input_dim, output_dim, k=1)
        self.pre_relation = Mask_Matrix(num_sensors, moving_window, decay)


    def forward(self, input):
        ## input size (bs, scale_num, num_nodes, input_dim)
        ## output size (bs, output_node_t, output_node_s, output_dim)

        input_con = Conv_GraphST(input, self.moving_window, self.stride)
        ## input_con size (bs, num_windows, num_sensors, moving_window, feature_dim)
        bs, num_windows, num_sensors, moving_window, feature_dim = input_con.size()
        input_con_ = torch.transpose(input_con, 2, 3)
        input_con_ = torch.reshape(input_con_, [bs * num_windows, moving_window * num_sensors, feature_dim])

        A_input = self.graph_construction(input_con_)
        A_input = A_input * self.pre_relation

        X_output = self.MPNN(input_con_, A_input)
        X_output = torch.reshape(X_output, [bs, num_windows, moving_window, num_sensors, self.output_dim])
        X_output = torch.reshape(X_output, [bs, num_windows*moving_window, num_sensors, self.output_dim])

        return X_output


class GraphMPNN_block(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, scale_num, decay):
        super(GraphMPNN_block, self).__init__()

        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        self.MPNN = MPNN_mk(input_dim, output_dim, k=1)
        self.pre_relation = Mask_Matrix(num_sensors,scale_num,decay)

    def forward(self, input):
        bs, tlen, num_sensors, feature_dim = input.size()
        input_con_ = torch.reshape(input, [bs, tlen*num_sensors, feature_dim])

        A_input = self.graph_construction(input_con_)
        A_input = A_input*self.pre_relation

        X_output = self.MPNN(input_con_, A_input)
        X_output = torch.reshape(X_output, [bs, tlen, num_sensors, -1])

        return X_output