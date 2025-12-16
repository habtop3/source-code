import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print(input.shape)
        # print(self.weight.shape)
        support = torch.mm(input, self.weight)
        # print(support.shape)
        # print(3)
        # print(support.shape)
        # input:n*in_feature
        # self.weight:in_features:out_features
        # support:n * out_features
        output = torch.spmm(adj, support)
        # adj:n*n
        # support:n * out_features
        # output:n * out_features
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
# cuda:true
        # graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        # # print(graph)
        # # print(features.shape)
        # # print(edge_types)
        # # return self.graph（大图）,
        # # features.cuda(device=device), 一个批次图的节点特征
        # # edge_types.cuda(device=device)一个批次图的边类型
        # graph = graph.to(torch.device('cuda:0'))
        # features = features.to(torch.device('cuda:0'))
        # # features0 = features
        # edge_types = edge_types.to(torch.device('cuda:0'))

        # for conv in self.gtn:
        #     features = conv(graph, features, edge_types)
        # feature.shape : torch.Size([6028, 100])

        # features1 = self.ggnn(graph,features0,edge_types)
        # # feature1.shape : torch.Size([6028, 100])
        # x_i = batch.de_batchify_graphs(features0)
        # # print(f'x_i的形状:{x_i.shape}')
        # # x_i.shape:torch.size([64,426,100])
        # h_i = batch.de_batchify_graphs(features1)
        # # print(f'h_i的形状:{h_i.shape}')
        # # h_i.shape:torch.size([64,426,100])
        # c_i = torch.cat((h_i,x_i),dim=-1)
        # # print(f'c_i的形状:{c_i.shape}')
        # # c_i.shape:torch.size([64,426,200])
        # x = self.conv_l1(h_i.transpose(1,2))
        # # print(f'x的形状:{x.shape}')
        # x = self.batchnorm_1d(x)
        # # print(f'x的形状:{x.shape}')
        # x = f.relu(x)
        # # print(f'x的形状:{x.shape}')
        # x = self.maxpool1(x)
        # # print(f'x的形状:{x.shape}')
        # y = self.conv_l2(x)
        # # print(f'y的形状:{y.shape}')
        # y = self.batchnorm_1d(y)
        # # print(f'y的形状:{y.shape}')
        # y = f.relu(y)
        # # print(f'y的形状:{y.shape}')
        # y = self.maxpool1(y)
        # # print(f'y的形状:{y.shape}')
        # y = y.transpose(1,2)
        # z = self.conv_l1_for_concat(c_i.transpose(1,2))
        # # print(f'z的形状:{z.shape}')
        # z = self.batchnorm_1d_for_concat(z)
        # # print(f'z的形状:{z.shape}')
        # z = f.relu(z)
        # # print(f'z的形状:{z.shape}')
        # z = self.maxpool1_for_conact(z)
        # # print(f'z的形状:{z.shape}')
        # z_1 = z
        # z_2 = self.conv_l2_for_concat(z_1)
        # # print(f'z_2的形状:{z_2.shape}')
        # z_2 = self.batchnorm_1d_for_concat(z_2)
        # # print(f'z_2的形状:{z_2.shape}')
        # z_2 = f.relu(z_2)
        # # print(f'z_2的形状:{z_2.shape}')
        # z_2 = self.maxpool2_for_conact(z_2)
        # # print(f'z_2的形状:{z_2.shape}')
        # z_2 = z_2.transpose(1,2)
        # # print(f'z_2的形状:{z_2.shape}')
        # y = self.mlp_y(y)
        # # print(f'y:{y.shape}')
        # z = self.mlp_z(z_2)
        # # print(f'z:{z.shape}')
        # before_avg = torch.mul(y,z)
        # # print(f'before_avg的形状:{before_avg.shape}')
        # avg = before_avg.mean(dim=1)
        # # Y_1 = self.maxpool1(
        # #     f.relu(
        # #         self.batchnorm_1d(
        # #             self.conv_l1(h_i.transponse(1,2))
        # #         )
        # #     )
        # # )
        # graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        # print(graph)
        # print(features.shape)
        # print(edge_types)
        # return
        # self.graph（大图）,
        # features.cuda(device=device), 一个批次图的节点特征
        # edge_types.cuda(device=device)一个批次图的边类型
        # print(f'zituhshuliang{batch.num_of_subgraphs}')
        # print(f'{batch.graph}')
        # print(f'{batch.number_of_nodes}')
        # print(f'{batch.graphid_to_nodeids}')
        # print(f'{batch.graph.ndata}')
        # print(f'{batch.graph.ndata["features"]}')
        # print(f'{batch.graph.ndata["features"].shape}')
        # print(f'{type(batch.graph.ndata["features"])}')
        # print(f'{type(batch.graph.ndata)}')
        # feature = batch.graph.ndata['features']
        # featurelist = []
        # for _,v in batch.graphid_to_nodeids.items():
        #     v1 = v.cpu()
        #     indices = v1.numpy()
        #     l = len(indices)
        #     s = indices[0]
        #     end = indices[l-1]
        #     feature0 = feature[s:end+1]
        #     feature1 = nearest_neighbors(feature0,K=4,metric='minkowski')
        #     print(feature1.shape)
        #     featurelist.append(feature1)
        # print( '#' * 100)
        # print(len(featurelist))
        # print(type(featurelist[0]))# features0 = features.numpy()
        #         # features1 = nearest_neighbors(features0, K=4, metric='minkowski')
        #         # print('#' * 100)
        #         # print(type(features))
        #         # features = nearest_neighbors(features, K=4, metric='minkowski')
        #         # #####     channel 1
        #         # features0 = features
        #         # features1 = features0.to(torch.device('cuda:0'))
        #         # features2 = nearest_neighbors(X=features1, K=4, metric='minkowski')
        #         # features3 = features2.to(torch.device('cuda:0'))
        #         ####   channel2
