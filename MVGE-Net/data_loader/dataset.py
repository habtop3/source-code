import copy
import json
import sys
import os

os.chdir(sys.path[0])
import torch
from dgl import DGLGraph
from utils import load_default_identifiers, initialize_batch, debug
from data_loader.batch_graph import GGNNBatchGraph
# from .test import nearest_neighbors, diagonal_concat
import dgl
from tqdm import *
from scipy.sparse import coo_matrix
import networkx as nx
import numpy as np


##for each function
# n_ident=args.node_tag  node_features
# g_ident=args.graph_tag  graph
# l_ident=args.label_tag target
# example = DataEntry(datset=self   #Dataset,
#                     num_nodes=len(entry[self.n_ident]),
#                     features=entry[self.n_ident],
#                     edges=entry[self.g_ident],
#                     target=entry[self.l_ident][0][0])


class DataEntry:
    def __init__(self, datset, num_nodes, features, edges, target):
        self.dataset = datset  # Dateset
        self.num_nodes = num_nodes  # len(entry[self.n_ident])
        self.target = target  # entry[self.l_ident][0][0])
        self.graph = DGLGraph()
        self.features = torch.FloatTensor(features)  # self.features：num_nodes*100
        self.graph.add_nodes(self.num_nodes, data={'features': self.features})
        for s, _type, t in edges:  # 遍历所有边
            etype_number = self.dataset.get_edge_type_number(_type)  # 将边的类型用数字表示
            self.graph.add_edges(s, t, data={'etype': torch.LongTensor([etype_number])})  # 把边的信息融入图中


# dataset = DataSet(train_src=os.path.join(input_dir, 'devign-train-v2.json'),
#                   valid_src=os.path.join(input_dir, 'devign-valid-v2.json'),
#                   test_src=os.path.join(input_dir, 'devign-test-v2.json'),
#                   batch_size=args.batch_size #64,
#                   n_ident=args.node_tag  node_features,
#                   g_ident=args.graph_tag  graph,
#                   l_ident=args.label_tag target)
#     def get_edge_type_number(self, _type):
#         if _type not in self.edge_types:
#             self.edge_types[_type] = self.max_etype
#             self.max_etype += 1
#         return self.edge_types[_type]
class DataSet:
    def __init__(self, train_src, valid_src, test_src, batch_size, n_ident=None, g_ident=None, l_ident=None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.train_matrix = []
        self.valid_matrix = []
        self.test_matrix = []
        self.batch_size = batch_size  # 64
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.n_ident, self.g_ident, self.l_ident = load_default_identifiers(n_ident, g_ident, l_ident)
        self.read_dataset(train_src, valid_src, test_src)
        self.initialize_dataset()
        self.train_batch_graphs = []
        self.valid_batch_graphs = []
        self.test_batch_graphs = []
        self.initialize_sadj()

    #

    def initialize_sadj(self):
        self.load_train_sadj(self.train_batches, self.train_examples, self.train_matrix)
        self.load_valid_sadj(self.valid_batches, self.valid_examples, self.valid_matrix)
        self.load_test_sadj(self.test_batches, self.test_examples, self.test_matrix)

    def load_train_sadj(self, train_batches, train_examples, train_matrix):
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        for ids in tqdm(train_batches):
            train_datas = [train_examples[i] for i in ids]
            batch_graph = GGNNBatchGraph()
            for train_data in train_datas:  # entry:example
                batch_graph.add_subgraph(copy.deepcopy(train_data.graph))
            # features = batch_graph.graph.ndata['features']
            # sadj = nearest_neighbors(X=features, K=3, metric='minkowski')#矩阵中只有0/1
            # sadj = torch.tensor(sadj)
            # sadj = coo_matrix(sadj)
            train_matrixs = [train_matrix[i] for i in ids]
            sadj = diagonal_concat(train_matrixs)
            sadj_tensor = torch.tensor(sadj)
            sadj_tensor = coo_matrix(sadj_tensor)
            sadj_dgl = dgl.from_scipy(sadj_tensor, eweight_name="w")  # jiade
            batch_graph.g = sadj_dgl
            c1 = dgl.to_networkx(sadj_dgl)
            c1_np = np.array(nx.adjacency_matrix(c1).todense())
            c2 = batch_graph.graph
            c2 = dgl.to_networkx(c2)
            c2_np = np.array(nx.adjacency_matrix(c2).todense())
            c_np = c1_np & c2_np
            c_tensor = torch.tensor(c_np)
            c_tensor = coo_matrix(c_tensor)
            c_dgl = dgl.from_scipy(c_tensor)
            c_dgl = dgl.add_self_loop(c_dgl)
            batch_graph.g_c = c_dgl
            self.train_batch_graphs.append(batch_graph)

    def load_valid_sadj(self, valid_batches, valid_examples, valid_matrix):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        for ids in tqdm(valid_batches):
            valid_datas = [valid_examples[i] for i in ids]
            batch_graph = GGNNBatchGraph()
            for valid_data in valid_datas:  # entry:example
                batch_graph.add_subgraph(copy.deepcopy(valid_data.graph))
            features = batch_graph.graph.ndata['features']
            # sadj = nearest_neighbors(X=features, K=3, metric='minkowski')
            # sadj = torch.tensor(sadj)
            # sadj = coo_matrix(sadj)
            valid_matrixs = [valid_matrix[i] for i in ids]
            sadj = diagonal_concat(valid_matrixs)
            sadj_tensor = torch.tensor(sadj)
            sadj_tensor = coo_matrix(sadj_tensor)
            sadj_dgl = dgl.from_scipy(sadj_tensor, eweight_name="w")  # jiade
            batch_graph.g = sadj_dgl
            c1 = dgl.to_networkx(sadj_dgl)
            c1_np = np.array(nx.adjacency_matrix(c1).todense())
            c2 = batch_graph.graph
            c2 = dgl.to_networkx(c2)
            c2_np = np.array(nx.adjacency_matrix(c2).todense())
            c_np = c1_np & c2_np
            c_tensor = torch.tensor(c_np)
            c_tensor = coo_matrix(c_tensor)
            c_dgl = dgl.from_scipy(c_tensor)
            c_dgl = dgl.add_self_loop(c_dgl)
            batch_graph.g_c = c_dgl
            self.valid_batch_graphs.append(batch_graph)

    def load_test_sadj(self, test_batches, test_examples, test_matrix):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        for ids in tqdm(test_batches):
            test_datas = [test_examples[i] for i in ids]
            batch_graph = GGNNBatchGraph()
            for test_data in test_datas:  # entry:example
                batch_graph.add_subgraph(copy.deepcopy(test_data.graph))
            features = batch_graph.graph.ndata['features']
            # sadj = nearest_neighbors(X=features, K=3, metric='minkowski')
            # sadj = torch.tensor(sadj)
            # sadj = coo_matrix(sadj)
            test_matrixs = [test_matrix[i] for i in ids]
            sadj = diagonal_concat(test_matrixs)
            sadj_tensor = torch.tensor(sadj)
            sadj_tensor = coo_matrix(sadj_tensor)
            sadj_dgl = dgl.from_scipy(sadj_tensor)
            batch_graph.g = sadj_dgl
            c1 = dgl.to_networkx(sadj_dgl)
            c1_np = np.array(nx.adjacency_matrix(c1).todense())
            c2 = batch_graph.graph
            c2 = dgl.to_networkx(c2)
            c2_np = np.array(nx.adjacency_matrix(c2).todense())
            c_np = c1_np & c2_np
            c_tensor = torch.tensor(c_np)
            c_tensor = coo_matrix(c_tensor)
            c_dgl = dgl.from_scipy(c_tensor)
            c_dgl = dgl.add_self_loop(c_dgl)
            batch_graph.g_c = c_dgl
            self.test_batch_graphs.append(batch_graph)

    def initialize_dataset(self):
        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset(self, train_src, valid_src, test_src):
        if train_src is not None:
            debug('Reading Train File!')
            with open(train_src, "r") as fp:  # devign-train-v0.json  node_feature graph target
                train_data = []
                train_data = json.load(fp)  # train_data:list train_data[i]:dict
                for entry in tqdm(train_data):  # 一个entry对应一个源代码 node_features graph target
                    # example就是图
                    example = DataEntry(datset=self,
                                        num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident],
                                        target=entry[self.l_ident][0][0])
                    if self.feature_size == 0:
                        self.feature_size = example.features.size(1)  # 100
                        debug('Feature Size %d' % self.feature_size)
                    self.train_examples.append(example)
                    train_features = entry[self.n_ident]
                    try:
                        train_f = nearest_neighbors(X=train_features, K=1, metric='minkowski')  # 矩阵
                        print(train_f.shape)
                    except:
                        train_f = np.eye(len(entry[self.n_ident]))
                    self.train_matrix.append(train_f)
        if valid_src is not None:
            debug('Reading Validation File!')
            with open(valid_src, "r") as fp:
                valid_data = []
                valid_data = json.load(fp)
                for entry in tqdm(valid_data):
                    example = DataEntry(datset=self,
                                        num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident],
                                        target=entry[self.l_ident][0][0])
                    self.valid_examples.append(example)
                    valid_features = entry[self.n_ident]
                    try:
                        valid_f = nearest_neighbors(X=valid_features, K=1, metric='minkowski')  # 矩阵
                    except:
                        valid_f = np.eye(len(entry[self.n_ident]))
                    self.valid_matrix.append(valid_f)
        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src) as fp:
                test_data = []
                test_data = json.load(fp)
                for entry in tqdm(test_data):
                    example = DataEntry(datset=self,
                                        num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident],
                                        target=entry[self.l_ident][0][0])
                    self.test_examples.append(example)
                    test_features = entry[self.n_ident]
                    try:
                        test_f = nearest_neighbors(X=test_features, K=1, metric='minkowski')  # 矩阵
                    except:
                        test_f = np.eye(len(entry[self.n_ident]))
                    self.test_matrix.append(test_f)

    # self.edge_types = {}
    # self.max_etype = 0
    def get_edge_type_number(self, _type):  # _type 1,2,3,4,5
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=False)  # 返回列表 列表元素是数组
        return len(self.train_batches)  # 几个batch
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size, shuffle=False)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size, shuffle=False)

        return len(self.test_batches)
        pass

    # graph, targets = dataset.get_next_train_batch()  # first
    # def get_next_train_batch(self):
    #     if len(self.train_batches) == 0:
    #         self.initialize_train_batch()
    #     ids = self.train_batches.pop()#train_batches:索引列表的列表
    #     return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)
    # def get_dataset_by_ids_for_GGNN(self, entries, ids):
    #     taken_entries = [entries[i] for i in ids]
    #     #entries:train_examples
    #     #ids:一个批次的列表
    #     #entries[i]:example DataEntry
    #     #taken_entries 列表 存放example
    #     # 取数据 node_features,graph,target
    #     labels = [e.target for e in taken_entries]#取标签
    #     batch_graph = GGNNBatchGraph()
    #     for entry in taken_entries:#entry:example
    #         batch_graph.add_subgraph(copy.deepcopy(entry.graph))
    #     return batch_graph, torch.FloatTensor(labels)

    # def get_dataset_by_ids_for_GGNN(self, entries, ids):
    #     taken_entries = [entries[i] for i in ids]
    #     # entries:train_examples
    #     # ids:一个批次的列表
    #     # entries[i]:example DataEntry
    #     # taken_entries 列表 存放example
    #     # 取数据 node_features,graph,target
    #     labels = [e.target for e in taken_entries]  # 取标签
    #     batch_graph = GGNNBatchGraph()
    #     for entry in taken_entries:  # entry:example
    #         batch_graph.add_subgraph(copy.deepcopy(entry.graph))
    #     features = batch_graph.graph.ndata['features']
    #     sadj = nearest_neighbors(X=features,K=4 ,metric='minkowski')
    #     sadj = coo_matrix(sadj)
    #     sadj_dgl = dgl.from_scipy(sadj)
    #     batch_graph.g = sadj_dgl
    #     return  batch_graph,torch.FloatTensor(labels)

    # print('#'*100)
    # print(batch_graph.graph)
    # print(batch_graph.graph.ndata['features'].shape)
    # features = batch_graph.graph.ndata['features']
    # # print(features)
    # if len(features_list) == 0:
    #     features_list.append(features)
    #
    # for i in features_list:
    #     if torch.equal(features,i):
    #         continue
    #     else:
    #         features_list.append(features)
    #         for k in range(10):
    #             try:
    #                 sadj =  nearest_neighbors(X=features,K=k,metric='minkowski')
    #                 sadj = torch.tensor(sadj)
    #                 sparse_matrix = coo_matrix(sadj)
    #                 # print('&'*100)
    #                 # print(sparse_matrix.shape)
    #                 sadj_dgl = dgl.from_scipy(sparse_matrix)
    #                 dataset[k] = sadj_dgl
    #                 print(f'k={k}符合要求')
    #             except:
    #                 print(f'k={k}不符合要求')

    # graph, targets = dataset.get_next_train_batch()  # first
    # def get_next_train_batch(self):
    #     if len(self.train_batches) == 0:
    #         self.initialize_train_batch()
    #     ids = self.train_batches.pop()#train_batches:索引列表的列表 train_batches[[1,2,3],[4,5,6]]
    #     return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)
    def get_next_train_batch(self, step_count):
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        if len(self.train_batch_graphs) == 0:
            self.load_train_sadj(train_batches=self.train_batches, train_examples=self.train_examples)
        # print(len(self.train_batch_graphs))
        # print(len(self.train_batches))
        step_count = len(self.train_batches) - step_count - 1
        ids = self.train_batches[step_count % len(self.train_batches)]
        batch = self.train_batch_graphs[step_count % len(self.train_batches)]
        # ids = self.train_batches.pop()#train_batches:索引列表的列表 train_batches[[1,2,3],[4,5,6]]
        # batch = self.train_batch_graphs.pop()
        # print(len(self.train_batch_graphs))
        # print(len(self.train_batches))
        labels = self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)
        # print(len(batch.graphid_to_nodeids))
        # print(labels.size())
        return batch, labels

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        return torch.FloatTensor(labels)

    def get_next_valid_batch(self, step_count):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        if len(self.train_batch_graphs) == 0:
            self.load_valid_sadj(valid_batches=self.valid_batches, valid_examples=self.valid_examples)
        # ids = self.valid_batches.pop()  # train_batches:索引列表的列表 train_batches[[1,2,3],[4,5,6]]
        # batch = self.valid_batch_graphs.pop()
        ids = self.valid_batches[step_count % len(self.valid_batches)]
        batch = self.valid_batch_graphs[step_count % len(self.valid_batches)]
        labels = self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)
        return batch, labels

    def get_next_test_batch(self, step_count):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        if len(self.train_batch_graphs) == 0:
            self.load_test_sadj(test_batches=self.test_batches, test_examples=self.test_examples)
        ids = self.test_batches[step_count % len(self.test_batches)]
        batch = self.test_batch_graphs[step_count % len(self.test_batches)]
        # ids = self.test_batches.pop()  # train_batches:索引列表的列表 train_batches[[1,2,3],[4,5,6]]
        # batch = self.test_batch_graphs.pop()
        labels = self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)
        return batch, labels
