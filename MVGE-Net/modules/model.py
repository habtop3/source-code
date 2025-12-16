import torch
from dgl.nn.pytorch import GatedGraphConv
from torch import nn
import torch.nn.functional as F
from graph_transformer_layers import GraphTransformerLayer
from mlp_readout import MLPReadout
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, RelGraphConv
from data_processing.test import nearest_neighbors,euclidean_distance_matrix,cosine_similarity_matrix,transform_tensor,load_sadj_dgl
import dgl
import pickle
import json
from scipy.sparse import csr_matrix,coo_matrix,csc_matrix
def dict_to_graph(graph_dict):
    g = dgl.graph((graph_dict['edges']['src'], graph_dict['edges']['dst']))
    return g
class  Attention(nn.Module):
    def __init__(self,in_size,hidden_size=50):
        super(Attention,self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,1,bias=False)
        )
        self.lin1 = nn.Linear(in_size,hidden_size)
        self.act = nn.Tanh()
        self.lin2 = nn.Linear(hidden_size,1,bias=False)
    def forward(self,z):
        #z:8031 3 100
        w = self.project(z)
        # print(z)
        # w0 =self.lin1(z)
        # print('3'*100)
        # print(w0)
        # w1 = self.act
        # w = self.lin2(2)
        # print(7)
        # print(w)
        # print(w.shape)
        beta = torch.softmax(w,dim=1)
        # print(8)
        # print(beta.shape)
        return  (beta * z).sum(1),beta

class GCN(nn.Module):
    def __init__(self,in_features,hid_features,out_features,dropout):
        super(GCN,self).__init__()
        '''

        '''
        self.gc1 = GraphConv(in_features,hid_features,weight=True)#100 200
        self.gc2 = GraphConv(hid_features,out_features,weight=True)
        self.dropout= dropout
    def forward(self,g,f):
        # print(x.shape)
        # print(adj.shape)
        # print('$')
        '''
        fadj == 本身包含的adj benshen baohan de adj - dgl
        sadj ==  knn - n*n   ? dgl
        fadj ---- sadj
        sadj ---- fadj
        '''
        x = F.relu(self.gc1(g, f))
        # f:8031*100  gc1:100 200
        # x:8031*200
        x = F.dropout(x,self.dropout,training=self.training)
        # x:8031*200
        # gc2: 200*100
        x = self.gc2(g, x)
        # print(x)
        # x:8031*100
        return x
class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, small_kernel, large_kernel, stride, groups):
        super().__init__()
        #   in_channels:  200
        #   out_channels: 200
        #   small_kernel: 3
        #   large_kernel: 11
        #   strid       : 1
        #   groups      : 200
        self.large_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size= large_kernel, stride= stride, padding= large_kernel // 2, groups=groups, dilation=1, bias = True)
        self.large_bn = torch.nn.BatchNorm1d(out_channels)
        self.small_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size= small_kernel, stride= stride, padding= small_kernel // 2, groups=groups, dilation=1)
        self.small_bn = torch.nn.BatchNorm1d(out_channels)
    def forward(self, inputs):
        large_out = self.large_conv(inputs)
        large_out = self.large_bn(large_out)
        small_out = self.small_conv(inputs)
        small_out = self.small_bn(small_out)
        return large_out + small_out
# model = DevignModel(input_dim=dataset.feature_size 100 , output_dim=100,
#                             num_steps=args.num_steps  6  , max_edge_types=dataset.max_edge_type  5 )
class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8,dropout=0.5,nfeat=100,nhid1=200,nhid2=100,nclass=2):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim #100
        self.out_dim = output_dim#100
        self.max_edge_types = max_edge_types#5
        self.num_timesteps = num_steps#6
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.gcn = GraphConv(in_feats=input_dim, out_feats=output_dim)
        n_layers = 3
        num_head = 10
        self.n_layers = n_layers
        self.gtn = nn.ModuleList([GraphTransformerLayer(input_dim=input_dim,
                                                         output_dim=output_dim,
                                                         num_heads = num_head,
                                                        dropout = 0.2,
                                                        max_edge_types = max_edge_types,
                                                         layer_norm= False,
                                                         batch_norm= True,
                                                         residual= True)
                                                        for _ in range (n_layers - 1)]  )
        self.MPL_layer = MLPReadout(output_dim, 2)


        ffn_ratio = 2
        self.concat_dim = output_dim#100
        small_kernel = 3
        large_kernel = 11
        self.RepLK = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.concat_dim),
            torch.nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, kernel_size=1, stride=1, padding=0, groups=1, dilation=1),
            torch.nn.ReLU(),
            ReparamLargeKernelConv(in_channels = self.concat_dim * ffn_ratio, out_channels = self.concat_dim * ffn_ratio, small_kernel = small_kernel, large_kernel = large_kernel, stride = 1, groups = self.concat_dim * ffn_ratio),
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.concat_dim * ffn_ratio, self.concat_dim, kernel_size=1, stride=1, padding=0, groups=1, dilation=1),
        )
        k = 3
        self.Avgpool1 =  torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(k, stride= k),
            torch.nn.Dropout(0.1)
        )
        self.ConvFFN = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.concat_dim),
            torch.nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, kernel_size=1, stride=1, padding=0, groups=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(self.concat_dim  * ffn_ratio, self.concat_dim, kernel_size=1, stride=1, padding=0, groups=1),
        )
        self.Avgpool2 =  torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(k, stride= k),
            torch.nn.Dropout(0.1)
        )

        # self.maxpool1 = torch.nn.MaxPool1d()
        self.conv_l1 = torch.nn.Conv1d(self.concat_dim,self.concat_dim,3)
        self.conv_l2 = torch.nn.Conv1d(self.concat_dim,self.concat_dim,3)
        self.batchnorm_1d = torch.nn.BatchNorm1d(output_dim)
        self.maxpool1 = torch.nn.MaxPool1d(3,stride=2)
        self.maxpool2 = torch.nn.MaxPool1d(3,stride=2)
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim * ffn_ratio, self.concat_dim*ffn_ratio, 3)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim*ffn_ratio, self.concat_dim*ffn_ratio, 3)
        self.batchnorm_1d_for_concat = torch.nn.BatchNorm1d(output_dim*ffn_ratio)
        self.maxpool1_for_conact = torch.nn.MaxPool1d(3, stride=2)
        self.maxpool2_for_conact = torch.nn.MaxPool1d(3, stride=2)
        self.mlp_y = nn.Linear(in_features=self.concat_dim,out_features=2)
        self.mlp_z = nn.Linear(in_features=self.concat_dim*ffn_ratio,out_features=2)
        self.sigmoid = nn.Sigmoid()
        # emb1 = self.SGCN1(features1, sadj)
        self.SGCN1  = GCN(in_features=nfeat, #100
                          hid_features=nhid1,#200
                          out_features=nhid2,#100
                          dropout=dropout)
        self.SGCN2 = GCN(in_features=nfeat,#100
                         hid_features=nhid1,#200
                         out_features=nhid2,#100
                         dropout=dropout)
        self.CGCN  = GCN(in_features=nfeat,#100
                         hid_features=nhid1,#200
                         out_features=nhid2,#100
                         dropout=dropout)
        self.dropout = dropout
        self.attention = Attention(nhid2)
        self.MLP = nn.Sequential(
            nn.Linear(nfeat,nclass),
            nn.Softmax(dim=1)
        )
        self.lin = nn.Linear(nfeat,nclass)
        self.act = nn.LogSoftmax(dim=1)
        # predictions = model(graph, cuda=True)
        # graph batch_graph：GGNNBatchGraph() 一个大图
    def forward(self, batch,cuda=False):
        '''
        sadj knn
        fadj 本身
        '''
        # sadj = batch.graph.ndata['sadj']
        # sparse_mat1 = coo_matrix(sadj)
        # # print(type(sparse_mat1))
        # g = dgl.from_scipy(sparse_mat1)
        # dgl_json = '/home/hab/Downloads/code/AMPLE-main/AMPLE_code/dgl.json'
        # with open(dgl_json, 'r') as f:
        #     data = json.load(f)
        # for dic in data:
        #     a0 = batch.graph.num_nodes()
        #     a1 = batch.graph.num_edges()
        #     a = int(str(a0) + str(a1))
        #     b = int(list(dic.keys())[0])
        #     if a == b:
        #         graph_dict = dic[list(dic.keys())[0]]
        #         dgl_dict = graph_dict['1']
        #         dgl = dict_to_graph(dgl_dict)
        # features = batch.graph.ndata['features']
        # sadj = nearest_neighbors(X=features,K=1,metric='minkowski')
        # sadj = coo_matrix(sadj)
        # sadj_dgl = dgl.from_scipy(sadj)
        # dgl_bin = './dgl.bin'
        # with open(dgl_bin, 'rb') as f:
        #     data = pickle.load(f)
        #     f.close()
        # a0 = batch.graph.num_nodes()
        # a1 = batch.graph.num_edges()
        # a = int(str(a0) + str(a1))
        # for dic in data:
        #     # print(a)
        #     b = int(list(dic.keys())[0])
        #     # print(b)
        #     if a == b:
        #         sadj_dgl = dic[a][1]
        #     else:
        #         dic = {}
        #         knn_dict = load_sadj_dgl(batch)
        #         dic[a] = knn_dict
        #         with open(dgl_bin, 'rb') as file:
        #             data = pickle.load(file)
        #             file.close()
        #         if isinstance(data, list):
        #             #             # 新增的字典
        #             new_item = dic
        #             #             # 向列表中添加新的字典
        #             data.append(new_item)
        #         else:
        #             print("JSON 数据格式不正确，期望列表类型")
        #         with open(dgl_bin, 'wb') as file1:
        #             pickle.dump(data, file1)
        #             file1.close()
        #         sadj_dgl = knn_dict[1]
        g = batch.g #特征空间
        g = g.to(torch.device('cuda:0'))
        f = batch.graph.ndata['features']
        f = f.to(torch.device('cuda:0'))
        g1 = batch.graph
        g1 = g1.to(torch.device('cuda:0'))
        emb1 = self.SGCN1(g,f)  #特征 a
        # print(emb1)
        # print(emb1.shape)
        com1 = self.CGCN(g,f)   #共同a
        # print(com1)
        # print(com1.shape)
        com2 = self.CGCN(g1,f)  #共同b
        # print(com2)
        # print(com2.shape)
        emb2 = self.SGCN2(g1,f) #拓扑b
        # print(emb2)
        # print(emb2.shape)
        xcom = (com1+com2)/2   #拓扑
        emb = torch.stack([emb1,emb2,xcom],dim=1)
        emb ,att =self.attention(emb)
        # print(emb.shape)
        # print(emb)
        # print(att.shape)
        # print(att)
        output = self.MLP(emb)
        # print(output.shape)
        # print(output)
        graph_features = []
        for k,v in batch.graphid_to_nodeids.items():
            # print(type(v))
            v = v.cpu()
            v = v.numpy()
            # print(type(v))
            s = v[0]
            e = v[-1]
            graph_feature = output[s:e]
            # print(graph_feature)
            graph_feature = transform_tensor(graph_feature)
            # print(graph_feature)
            # print(graph_feature.shape)
            graph_features.append(graph_feature)
        concat_feature = torch.cat(graph_features,dim=0)
        concat_feature = concat_feature.to(torch.device('cuda:0'))





        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph = graph.to(torch.device('cuda:0'))
        features = features.to(torch.device('cuda:0'))
        edge_types = edge_types.to(torch.device('cuda:0'))
        for conv in self.gtn:
            features = conv(graph, features, edge_types)
            # print(graph, graph.shape)
        # feature.shape : torch.Size([6028, 100])
        outputs = batch.de_batchify_graphs(features)
        # outputs.shape:torch.size([64,426,100])
        # 64:批次大小
        # 426:该批次图中节点最多的图的结点数
        # 100:批次图单点的特征维度:{c}')
        outputs = outputs.transpose(1, 2)
        # print(outputs.shape)
        ''' 
              Layer1
        '''
        outputs += self.RepLK(outputs)
        # print(outputs.shape)
        outputs = self.Avgpool1(outputs)
        # print(outputs.shape)
        outputs += self.ConvFFN(outputs)
        # self.ConvFFN = torch.nn.Sequential(
        #     torch.nn.BatchNorm1d(self.concat_dim),
        #     torch.nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, kernel_size=1, stride=1, padding=0, groups=1),
        #     torch.nn.GELU(),
        #     torch.nn.Conv1d(self.concat_dim * ffn_ratio, self.concat_dim, kernel_size=1, stride=1, padding=0, groups=1),
        # )
        # print(outputs.shape)
        outputs = self.Avgpool2(outputs)
        # print(outputs.shape)
        # self.Avgpool2 = torch.nn.Sequential(
        #     torch.nn.ReLU(),
        #     torch.nn.AvgPool1d(k, stride=k),
        #     torch.nn.Dropout(0.1)
        # )
        '''
              Layer2
        ''' 
        outputs = outputs.transpose(1, 2)
        # print(outputs.shape)
        # print(outputs.sum(dim=1).shape)
        outputs = self.MPL_layer(outputs.sum(dim=1))
        # print(100)
        # print(outputs)
        # print(outputs.shape)
        # outputs = outputs + avg
        # self.MPL_layer = MLPReadout(output_dim, 2)
        # outputs = self.sigmoid(outputs)
        outputs = nn.Softmax(dim=1)(outputs)
        # print('$'*100)
        # print(outputs)
        # print(outputs.shape)
        # # print(outputs)
        # # print(outputs.shape)
        outputs = (concat_feature + outputs) / 2
        print("output",outputs)
        return outputs


