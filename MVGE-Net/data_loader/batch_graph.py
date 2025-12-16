import torch
from dgl import DGLGraph
import numpy
import dgl
import dgl.function as fn
from torch import nn


class BatchGraph:
    def __init__(self):
        self.graph = DGLGraph()
        self.g = DGLGraph()
        self.number_of_nodes = 0
        self.graphid_to_nodeids = {}
        self.num_of_subgraphs = 0
        # 新增
        self.g_c = DGLGraph()

    def add_subgraph(self, _g):
        assert isinstance(_g, DGLGraph)
        num_new_nodes = _g.number_of_nodes()
        self.graphid_to_nodeids[self.num_of_subgraphs] = torch.LongTensor(
            list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes))).to(torch.device('cuda:0'))
        self.graph.add_nodes(num_new_nodes, data=_g.ndata)
        sources, dests = _g.all_edges()
        sources += self.number_of_nodes
        dests += self.number_of_nodes
        self.graph.add_edges(sources, dests, data=_g.edata)
        self.number_of_nodes += num_new_nodes
        self.num_of_subgraphs += 1

    # def add_subgraph(self, _g):
    #     assert isinstance(_g, DGLGraph)
    #     num_new_nodes = _g.number_of_nodes()#节点数
    #     self.graphid_to_nodeids[self.num_of_subgraphs] = torch.LongTensor(
    #         list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes))).to(torch.device('cuda:0'))
    #     # print(self.graphid_to_nodeids)
    #     # self.graph.add_nodes(num_new_nodes, data=_g.ndata)
    #     sources, dests = _g.all_edges()
    #     node_feature = _g.ndata['features']
    #     # print(f'node_feature值为{node_feature}')
    #     edge_type = _g.edata['etype']
    #     assert len(sources) == len(dests) and len(sources) == len(edge_type)
    #     source = sources.tolist()
    #     dest = dests.tolist()
    #     edge_type = edge_type.tolist()
    #     node = set(source + dest)
    #     node = list(node)
    #     count = {}
    #     for i in node:
    #         count[f'{i}'] = 0
    #     for i, e_t in enumerate(edge_type):
    #         if e_t == 1 or e_t == 4:
    #             continue
    #         else:
    #             count[f'{source[i]}'] += 1
    #             count[f'{dest[i]}'] += 1
    #     values = list(count.values())
    #     for i in range(node_feature.shape[0]):
    #         if values[i] == 0:
    #             node_feature[i] = node_feature[i]
    #         else:
    #             node_feature[i] = node_feature[i] * values[i]
    #     _g.ndata['features'] = node_feature
    #     # print(f'node_feature值为{node_feature}')
    #     self.graph.add_nodes(num_new_nodes, data=_g.ndata)
    #     sources += self.number_of_nodes
    #     dests += self.number_of_nodes
    #
    #     self.graph.add_edges(sources, dests, data=_g.edata)
    #
    #     self.number_of_nodes += num_new_nodes
    #
    #     self.num_of_subgraphs += 1

    def cuda(self, device=None):
        for k in self.graphid_to_nodeids.keys():
            self.graphid_to_nodeids[k] = self.graphid_to_nodeids[k].cuda(device=device)

    def de_batchify_graphs(self, features=None):
        assert isinstance(features, torch.Tensor)
        # vectors:列表，存放每个子图的节点特征（经过边加权）
        vectors = [features.index_select(dim=0, index=self.graphid_to_nodeids[gid]) for gid in
                   self.graphid_to_nodeids.keys()]  # 取一个大图中小图的特征
        # print(len(vectors))
        # print(vectors[0].shape)
        # lengths:每个子图的节点数量
        lengths = [f.size(0) for f in vectors]
        # print(lengths)
        count = 0
        for i in range(len(lengths)):
            count = count + lengths[i]
        # print(count)
        # max_len : 一个批次所有图中，节点数量最大的图的节点数量
        # max_len : 最大结点数
        max_len = max(lengths)
        # print(max_len)
        for i, v in enumerate(vectors):
            vectors[i] = torch.cat((v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])),
                                                   requires_grad=v.requires_grad, device=v.device)), dim=0)
        output_vectors = torch.stack(vectors)
        # print(output_vectors.shape)
        return output_vectors  # , lengths

    def get_network_inputs(self, cuda=False):
        raise NotImplementedError('Must be implemented by subclasses.')


from scipy import sparse as sp


class GGNNBatchGraph(BatchGraph):
    def __init__(self):
        super(GGNNBatchGraph, self).__init__()  # 调用父类的init方法
        # def __init__(self):
        #     self.graph = DGLGraph()
        #     self.number_of_nodes = 0
        #     self.graphid_to_nodeids = {}
        #     self.num_of_subgraphs = 0

    # graph, features, edge_types = batch.get_network_inputs(cuda=cuda) cuda:true
    def get_network_inputs(self, cuda=False, device=None):

        features = self.graph.ndata['features']  # 一个批次图的节点特征
        edge_types = self.graph.edata['etype']  # 一个批次图的边类型

        if cuda:
            return self.graph, features.cuda(device=device), edge_types.cuda(device=device)
        else:
            return self.graph, features, edge_types
        pass
