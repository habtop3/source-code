import torch
from dgl.nn.pytorch import GatedGraphConv
from torch import nn
import torch.nn.functional as f
import torch.nn.functional
from dgl.function import message
from dgl.function import reducer
from dgl.nn.pytorch import GatedGraphConv, GraphConv, AvgPooling, MaxPooling,  RelGraphConv
import dgl
import dgl.function as fn
from dgl.function import BuiltinFunction
import scipy.sparse as sp
import math
import numpy as np
from dgl.function.message import _gen_message_builtin
import sys



"""
    GraphNorm

"""
class Norm(nn.Module):

    def __init__(self, norm_type = 'gn', hidden_dim=100, print_info=None):
        super(Norm, self).__init__()
        # assert norm_type in ['bn', 'ln', 'gn', None]
        self.norm = None
        self.print_info = print_info
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
            # self.Graph_norm1 = Norm(hidden_dim=output_dim)
            # h = self.Graph_norm1(graph, h)
    def forward(self, graph, tensor, print_=False):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor
        tensor = tensor
        batch_list = graph.batch_num_nodes().float()#返回每个图节点的数量
        batch_size = len(batch_list)#
        batch_list = torch.tensor(batch_list, device = torch.device('cuda:0'))
        batch_list = batch_list.long()
        
        batch_index = torch.arange(batch_size, device = torch.device('cuda:0')).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:], device = torch.device('cuda:0'))
        
        
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        
        result = self.weight * sub / std + self.bias
        return result.to(torch.device('cuda:0'))

"""
    Util functions
"""


u_mul_e = fn.message._gen_message_builtin("u",'e','mul')
# print(u_mul_e)
reducer_sum = fn.reducer._gen_reduce_builtin("sum")
# print(reducer_sum)
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func
def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-10, 10))}
        #return {field: torch.exp((edges.data[field] / scale_constant))}
    return func

"""
    MultiHeadAttentionLayer Layer
    多头注意力机制

"""
# self.attention = MultiHeadAttentionLayer(input_dim  100, output_dim // num_heads  100/10, num_heads 10, use_bias)
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias = True):
        super().__init__()
        #in_dim:100
        #out_dim:10
        self.out_dim = out_dim #10
        self.num_heads = num_heads#10
        num_steps = 6
        max_edge_types = 5
        self.feature_Q = RelGraphConv(in_feat=in_dim, out_feat=in_dim, num_rels = 9, regularizer='basis', num_bases = 9, activation = f.relu, self_loop = False, dropout = 0.1)
        self.feature_K = RelGraphConv(in_feat=in_dim, out_feat=in_dim, num_rels = 9, regularizer='basis', num_bases = 9, activation = f.relu, self_loop = False, dropout = 0.1)
        self.feature_V = RelGraphConv(in_feat=in_dim, out_feat=in_dim, num_rels = 9, regularizer='basis', num_bases = 9, activation = f.relu, self_loop = False, dropout = 0.1)
    def propagate_attention(self, g):
        # Compute attention score
        #### 乘
        #### 这时的图的边没有score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)
        #   只是给图新增了一个score特征
        #   g.edata['score']:num_edges * 10 * 1
        #   torch.Size([19575，10，1]) 19575是这个批次组成的大图的边的数量
        #####放缩
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))
        #####放缩后score形状不变
        #   g.edata['score']:num_edges * 10 * 1
        #   score:torch.Size([19575，10，1]) 19575是这个批次组成的大图的边的数量
        # Send weighted values to target nodes
        eids = g.edges()
        # print(g)
        ####加权
        g.send_and_recv(eids, fn.u_mul_e('V_h','score','V_h'),fn.sum('V_h','wV'))
        #####
        # u_mul_e 只是用来产生消息，对应send 并不会改变V_h的值
        # reducer_sum 传递聚合消息对应recv 将消息聚合
        # print(g.ndata['wV'].shape)
        # wV:torch.Size([6028，10，10])
        # print(g)
        # a = g.ndata['wV']
        # b = g.ndata['V_h']
        # print(torch.equal(a,b))
        # print(f'g.ndata[V_h]通过边加权后的形状：{z}')
        # print(f'wV的形状{z1}')
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))
        # print(g)
        #Z:torch.Size([6028，10，1])
        # b = g.ndata['z'].shape
        # print(f'g.ndata[z]形状{b}')
    # attn_out = self.attention(graph, h, e)
    # self.attention = MultiHeadAttentionLayer(input_dim, output_dim // num_heads, num_heads, use_bias)
    # attn_out = self.attention(graph, h, e)
    def forward(self, g, h, e):
        # g:graph->self.graph:大图（这个批次所有图组合在一起）
        # h->features.cuda(device=device):大图的节点特征
        # e->edge_types.cuda(device=device):一个批次图的边类型
        '''
        对不同类型的边进行处理
        '''
        feature_Q = self.feature_Q(g, h, e)
        feature_K = self.feature_K(g, h, e)
        feature_V = self.feature_V(g, h, e)

        ########################################
        # feature_Q,feature_K,feature_V形状相同
        # torch.Size([6028, 100])

        Q_h = feature_Q
        K_h = feature_K
        V_h = feature_V

        ########################################
        # Q_h,K_h,V_h形状相同
        # torch.Size([6028, 100])

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        # Q_h:num_nodes *100
        # g.ndata['Q_h']:num_nodes*100 -> -1 * [num_heads(10) * out_dim(10)](100)
        # -1:num_nodes
        '''
        resize
        '''
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        ########################################
        # g.ndata['Q_h'],g.ndata['K_h'],g.ndata['V_h']形状相同
        # g.ndata[Q/K/V_h].shape:torch.Size([6028, 10,10])

        self.propagate_attention(g)

        #head_out = g.ndata['wV'] / g.ndata['z']
        head_out= g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6)) # adding eps to all values here

        return head_out
        # head_out.shape:torch.Size([6028, 10, 10])

"""
    Graph Transformer Layer

"""
class GraphTransformerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, input_dim, output_dim, max_edge_types, num_heads, num_steps=8, dropout=0.0, layer_norm=False, batch_norm=True, residual=False,
                 use_bias=True):
        super().__init__()
        self.in_channels = input_dim #100
        self.out_channels = output_dim #100
        self.num_heads = num_heads #10
        self.dropout = dropout #0.2
        self.residual = residual #true
        self.layer_norm = layer_norm #false
        self.batch_norm = batch_norm #true
        self.max_edge_types = max_edge_types #5
        self.num_timesteps = num_steps #6
        self.attention = MultiHeadAttentionLayer(input_dim, output_dim // num_heads, num_heads, use_bias)
        self.O = nn.Linear(output_dim, output_dim)
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(output_dim)
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(output_dim)
            self.Graph_norm1 = Norm(hidden_dim=output_dim)
        # FFN
        self.FFN_layer1 = nn.Linear(output_dim, output_dim * 2)
        self.FFN_layer2 = nn.Linear(output_dim * 2, output_dim)
        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(output_dim)
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
            self.Graph_norm2 = Norm(hidden_dim=output_dim)
    # graph->self.graph:大图（这个批次所有图组合在一起）
    # h->features.cuda(device=device):大图的节点特征
    # e->edge_types.cuda(device=device):一个批次图的边类型
    # features = conv(graph, features, edge_types)
    def forward(self, graph, h, e):
        '''
        graph:dgl
        h:feature
        e:type
        '''
        h_in1 = h
        # for first residual connection
        #_in1= h_in1.view(-1, self.out_channels)
        # multi-head attention out
        # batch_norm:true
        # layer_norm:false
        # print(f'送入graph_transformer的h形状{h.shape}')
        #  h.shape:torch.Size([6028, 100])
        if self.batch_norm: #true
            # self.Graph_norm1 = Norm(hidden_dim=output_dim)
            h = self.Graph_norm1(graph, h)
        # bubian
        # print(f'第一次batch_norm后的h形状{h.shape}')
        #  torch.Size([6028, 100])
            #return h：features
        #多头注意力机制网络 输入 graph 和 h ,仅使用单头，因为每个输入的维度不一样
        # print(f'送入注意力的h形状{h.shape}')
        #   h.shape:torch.Size([6028, 100])
        attn_out = self.attention(graph, h, e)
        # self.attention = MultiHeadAttentionLayer(input_dim, output_dim // num_heads, num_heads, use_bias)
        # print(attn_out.shape)
        #######
        #attn_out:torch.Size([6028,10,10])
        h = attn_out.view(-1, self.out_channels)
        # print(f'注意力后的h形状{h.shape}')
        # h:torch.Size([6028,100])
        h = f.dropout(h, self.dropout, training=self.training)
        #h.shape:torch.Size([6028, 100])
        #dense层 nn.Linear(output_dim, output_dim)
        h = self.O(h)
        # self.O = nn.Linear(output_dim, output_dim)
        # print(f'线性连接后的h形状{h.shape}')
        # h.shape:torch.Size([6028, 100])
        #是否使用残差连接 目前是矩阵模块未对齐
        if self.residual:#true
            h = h_in1 + h  # residual connection
        # h.shape: torch.Size([6028, 100])
        # print(f'残差连接后的h形状{h.shape}')
        if self.layer_norm:#false
            h = self.layer_norm1(h)
        # h.shape: torch.Size([6028, 100])
        h_in2 = h  # for second residual connection
        if self.batch_norm:#true
            #h = self.batch_norm2(h)
            h = self.Graph_norm2(graph, h)
            # self.Graph_norm1 = Norm(hidden_dim=output_dim)
        # print(f'第二次batch_norm后的h形状{h.shape}')
        # h.shape: torch.Size([6028, 100])
        # FFN
        # 连续的两个前向网络 两个前向网络后 输出的维度和输入相同
        h = self.FFN_layer1(h)
        # h: torch.Size([6028, 200])
        h = f.relu(h)
        h = f.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)
        # h: torch.Size([6028, 100])
        if self.residual:#true
            h = h_in2 + h  # residual connection
        # h: torch.Size([6028, 100])

        if self.layer_norm:#false
            h = self.layer_norm2(h)

        # h: torch.Size([6028, 100])


        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)
