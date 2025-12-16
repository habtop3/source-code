# coding=UTF-8
import argparse
import csv
from enum import unique
from genericpath import exists
from operator import index
#from numpy.core.fromnumeric import nonzero
import json
import numpy as np
import os
from gensim.models import Word2Vec
from tqdm import tqdm
import copy
import re
import nltk
import warnings
import pandas as pd
from utils import my_tokenizer, check
from transformers import AutoTokenizer, AutoModel
import torch

edgeType_reduced = {
    'IS_AST_PARENT': 1,
    'FLOWS_TO': 2,
    'REACHES': 3,
    'NSC': 4,
    'Self_loop': 5
}
ver_edge_type_reduced = {
    '1' : 'IS_AST_PARENT',
    '2' : 'FLOWS_TO',
    '3' : 'REACHES',
    '4' : 'NSC',
    '5' : 'Self_loop'
}

edge_type_ast = {
    'IS_AST_PARENT' : 'black',
    'FLOWS_TO': 'blue',
    'NSC': 'red'
}
allowed_edge_types_full = {
    'IS_AST_PARENT': 'black',
    'IS_CLASS_OF': 'purple',
    'FLOWS_TO': 'blue',
    'DEF': 'green',
    'USE': 'green',
    'REACHES': 'black',
    'CONTROLS': 'black',
    'DECLARES': 'green',  #
    'DOM': 'red',
    'POST_DOM': 'red',
    'IS_FUNCTION_OF_AST': 'green',
    'IS_FUNCTION_OF_CFG': 'green'
}
def read_csv(csv_file_path):#返回一个列表，列表的内容是字典，是csv文件中除了每一行内容对应的字典，键是表头，值是该行表头对应的值
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()#读表头
        header = header.strip()#去处表头两侧空白字符
        h_parts = [hp.strip() for hp in header.split('\t')]#得到表头各个字段  列表
        for line in fp:
            line = line.strip()#去除当前行两侧的空白字符
            instance = {}
            lparts = line.split('\t')#得到每个行数据的各个部分 返回列表
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data
# build_ast(nodeKey, edges, ast_edges)
def build_ast(starts, edges, ast_edges):#starts:列表,列表元素为一个csv文件的每行的key（是控制流图节点，不是控制流图输入输出节点）,edges:edges是一个列表，列表的内容是字典，是一个csv文件中除了第一行每一行内容对应的字典，字典的键该csv的是表头，值是该行表头对应 ast_edges:空列表
    if len(starts) == 0:#c语言文件中没有控制流图节点
        return
    new_starts = []
    for i in starts:#最开始starts只有一个key,每次传入一个控制流图节点
        ast = {}
        ast['start'] = i
        ast['end'] = []
        for edge in edges:
            if edge['start'].strip() == i and edge['type'].strip() == 'IS_AST_PARENT':#找到以控制流图节点为起点且边类型为ast边的节点
                ast['end'].append(edge['end'].strip())
                new_starts.append(edge['end'].strip())
        if len(ast['end']) > 0:
            ast_edges.append(ast)
    build_ast(new_starts, edges, ast_edges)
    pass
def get_nodes_by_key(nodes, key):
    for node in nodes:
        if node['key'].strip() == key:
            return node
    return  None
# nsc_edges = get_ncs_edges(all_ast_edges, 'NSC', nodes)
#  list[list[dict[str, list]]]
# all_ast_edges：列表 列表元素为列表，列表：列表内容为字典 每个字典对应每个node{'start' key, 'end' []}
# ast_sent:列表，列表内容为字典 每个字典对应每个node{'start' key, 'end' []}
# def get_nodes_by_key(nodes, key):
#     for node in nodes:
#         if node['key'].strip() == key:
#             return node
#     return  None
# nsc_edges = get_ncs_edges(all_ast_edges, 'NSC', nodes)
# all_ast_edges: list[list[dict[str, list]]] = []
# ast_edge列表，列表内容为字典 每个字典对应每个node{'start' key, 'end' []}
def get_ncs_edges(all_ast_edges, nsc_type, nodes):
    nsc_edges = list()
    first = True
    #sent_order_dup = sorted(set(sent_order), key = sent_order.index)
    par_sent = []
    tmp_sent = copy.deepcopy(all_ast_edges)
    # ast_edges:列表，列表元素为字典  例子：[{start:a end:[b,c]}{start:b end:[f]}]
    # ast_edges: 以当前控制流图节点为起点，记录 当前结点的key和它的ast连接节点的key
    # 每个控制流图节点一个ast_edges
    # ast_sent
    # ast_edges: 列表，列表元素为字典例子：[{start: a end: [b, c]}{start: b end: [f]}]
    for ast_sent in all_ast_edges:
        if get_nodes_by_key(nodes,ast_sent[0]['start'])['type'] == 'Parameter':
            par_sent.append(ast_sent)
            tmp_sent.remove(ast_sent)
    all_ast_edges = par_sent + tmp_sent  #改变顺序
    for ast_sent in all_ast_edges: #all_ast_edge: all csv ast_edge:一个csv
        s_nodes = []
        t_nodes = []
        for ast in ast_sent: #遍历一个csv中的每一行
            s_node = ast['start']
            s_nodes.append(s_node)
            t_nodes = t_nodes + ast['end']
        nsc_nodes = []
        if s_nodes == t_nodes and len(s_nodes) == 1:  # break; continue; return;
            nsc_nodes = t_nodes
        if len(s_nodes) == 1 and len(t_nodes) == 0:  # if(a) {start:'a', end:[]}
            nsc_nodes = s_nodes
        else:
            for node in t_nodes:
                if node not in s_nodes:
                    nsc_nodes.append(int(node))
        nsc_nodes.sort(reverse=False)#升序
        #print(nsc_nodes)
        idx = 0
        if first:
            first = False
            idx = 1
            s = nsc_nodes[0]
        for i in range(idx, len(nsc_nodes)):
            #if int(s) >= int(nsc_nodes[i]):
            #    ex = Exception("Sequence Order Error!!")
            #    raise(ex)
            edge = [str(s), nsc_type, str(nsc_nodes[i])]
            nsc_edges.append(edge)
            s = nsc_nodes[i]
    return nsc_edges

def spe_sent(n_type, n_code):
    if n_type == 'BreakStatement' and n_code == ['break', ';']:
        return True
    elif n_type == 'ContinueStatement' and n_code == ['continue', ';']:
        return True
    elif n_type == 'ReturnStatement' and n_code == ['return', ';']:
        return True
    elif n_type == 'InfiniteForNode' and n_code == ['true']:  #(;;)
        return True
    elif n_type == 'Label' and 'case' in n_code: #case 1:
        return True
    return False

edgeType_reduced = {
    'IS_AST_PARENT': 1,
    'FLOWS_TO': 2,
    'REACHES': 3,
    'NSC': 4,
    'Self_loop': 5
}
ver_edge_type_reduced = {
    '1' : 'IS_AST_PARENT',
    '2' : 'FLOWS_TO',
    '3' : 'REACHES',
    '4' : 'NSC',
    '5' : 'Self_loop'
}
# index_map, graph, nodes_num, edges_num = graphGeneration(nodes, edges, edgeType_reduced, ver_edge_type_reduced)
def  graphGeneration(nodes, edges, edge_type_map, ver_edge_type_map):#nodes，edges是列表，列表的元素是字典，是一个csv(c)文件中除了第一行每一行内容对应的字典，字典的键该csv的是表头，值是该行表头对应的值
    # 每个源函数调用一次graphGeneration
    # nodes,edges是一个csv文件(c文件)对应的列表，列表元素是字典,对应该csv文件的一行
    index_map = dict()
    index_map_ver = dict()
    all_nodes = set()
    all_ast_edges = []
    all_edges = []
    for node in nodes:#遍历列表 node:一个csv(c)文件的一行：字典
        if node['isCFGNode'].strip() != 'True' or node['type'].strip() == 'File':#保证是控制流图节点，type!=file
            continue
        all_nodes.add(node['key']) #all_nodes存储一个源代码(csv)中所有控制流图节点的key
        if node['type'] in ['CFGEntryNode','CFGExitNode']:#如果是控制流图的进入和退出节点
            continue
        nodeKey = [node['key']]#nodekey:列表 存储一个c语言文件控制流图(不包括控制流图出入节点)节点的key
        ast_edges = []  # 一个c文件中以其中一个控制流图节点为起点的ast边
        build_ast(nodeKey, edges, ast_edges)#每次传入一个控制流图节点
        # ast_edges:列表，列表元素为字典  例子：[{start:a end:[b,c]}{start:b end:[f]}]
        # ast_edges: 以当前控制流图节点为起点，记录当前结点的key和它的ast连接节点的key
        # 每个控制流图节点一个ast_edges
        if len(ast_edges) == 0:
            #break; continue; returns; (;;)
            if spe_sent(node['type'], node['code'].strip().split()):
                dic = {}
                dic['start'] = nodeKey[0]
                dic['end'] = nodeKey
                ast_edges.append(dic)
            else:
                return None, None, None, True
        all_ast_edges.append(ast_edges)
        # all_ast_edge对应一个csv,也就是对应一个.c中以所有控制流图节点为起点的ast
        # ast_edges:列表，列表元素为字典  例子：[{start:a end:[b,c]}{start:b end:[f]}]
        # list[list[dict[str, list]]] = []

    # nsc_edge
    nsc_edges = get_ncs_edges(all_ast_edges, 'NSC', nodes)#nsc:Natural Code Sequence
    # edge = [str(s), nsc_type, str(nsc_nodes[i])]
    # nsc_edges :list[list[str]]

    #all_edges中加入ast边
    ast_type = 'IS_AST_PARENT'
    for item in all_ast_edges:#遍历每个ast_edge 遍历每个控制流图节点的ast边
        # break; continue; return;
        if len(item) == 1 and len(item[0]['end']) == 1 and item[0]['start'] == item[0]['end'][0]:
            continue
        for x in item:#item:ast_edge x:字典 {start :int end[]}
            start = x['start']
            for end in x['end']:
                all_edges.append([start, ast_type, end]) #包含了ast边
    #all_edges中加入cfg,dfg边
    for e in edges:#edges一个csv文件对应的列表，e是该csv文件的每一行对应的字典
        start, end, eType = e['start'], e['end'], e['type']
        start_node = get_nodes_by_key(nodes, start)#返回字典  csv的某一行
        end_node = get_nodes_by_key(nodes, end)#返回字典 csv的某一行
        if start_node['isCFGNode'].strip() != 'True' or end_node['isCFGNode'].strip() != 'True':
            continue
        # 到这里 e就是连接控制流图中两个结点的边
        if eType != 'IS_FILE_OF' and eType != ast_type:
            if not eType in edge_type_map: #or not start in all_nodes or not end in all_nodes:
                continue
            all_edges.append([start, eType, end])
    # edgeType_reduced = {
    #     'IS_AST_PARENT': 1,
    #     'FLOWS_TO': 2,
    #     'REACHES': 3,
    #     'NSC': 4,
    #     'Self_loop': 5

    #all_edges 包括ast边和控制流图边,DFG all_edges list(list(str,str,str))
    for e in all_edges:
        start, _, end = e
        all_nodes.add(start)
        all_nodes.add(end)

    #all_nodes包括控制流图节点和ast边连接的节点
    if len(all_nodes) == 0 or len(all_nodes) > 500:
        print(node['location'])
        return None, None, None, None

    for i, node in enumerate(all_nodes):#all_nodes(2,3,6,9,45,67....)
        index_map[node] = i
        index_map_ver[i] = node

    all_edges_new = []   #original full graph
    #all_edge_new 有ast,cfg,dfg
    for e in all_edges: # e = [start, type不是数字, end]
        e_new = [index_map[e[0]], edge_type_map[e[1]], index_map[e[2]]]
        all_edges_new.append(e_new)#edge_eype为数字
    edgeType_reduced = {
        'IS_AST_PARENT': 1,
        'FLOWS_TO': 2,
        'REACHES': 3,
        'NSC': 4,
        'Self_loop': 5
    }

    ver_edge_type_reduced = {
        '1': 'IS_AST_PARENT',
        '2': 'FLOWS_TO',
        '3': 'REACHES',
        '4': 'NSC',
        '5': 'Self_loop'
    }
    #all_edge_new 有ast,cfg,nsc
    for e in nsc_edges:
        e_new = [index_map[e[0]], edge_type_map[e[1]], index_map[e[2]]]
        all_edges_new.append(e_new)

    #add self-loop
    loop = 'Self_loop'
    for node in all_nodes:
        self_loop = [index_map[node], edge_type_map[loop], index_map[node]]
        all_edges_new.append(self_loop)

    if len(all_edges_new) == 0:
        return None, None, None, None

    #统计不同边类型的数量
    edges_num = {}#etype:1
    for t in edge_type_map.keys():
        edges_num[t] = 0#etype:0
    for e in all_edges_new:
        key = ver_edge_type_map[str(e[1])]
        edges_num[key] += 1
    # ver_edge_type_reduced = {
    #     '1': 'IS_AST_PARENT',
    #     '2': 'FLOWS_TO',
    #     '3': 'REACHES',
    #     '4': 'NSC',
    #     '5': 'Self_loop'
    # }
    edges_num['nodes'] = len(all_nodes)

    return index_map_ver, all_edges_new, len(index_map_ver), edges_num
# index_map, graph, nodes_num, edges_num = graphGeneration(nodes, edges, edgeType_reduced, ver_edge_type_reduced)
# return index_map_ver, all_edges_new, len(index_map_ver), edges_num
# gInput = word2vec(nodes, index_map, graph, model)
# index_map[node] = i
# index_map_ver[i] = node
def word2vec(nodes, index_map, graph, wv):
    gInput = list()
    all_nodes = set()
    for item in graph: #all_edges_new
        s, _, e = item #start edge end
        all_nodes.add(e)
        all_nodes.add(s)
    if len(all_nodes) != len(index_map):#index_map index_map_ver
        return None
    for i in index_map:#index_map index_map_ver
        true_id = index_map[i]
        #print(true_id)
        node = get_nodes_by_key(nodes, true_id)
        node_content = node['code'].strip()#某个节点的代码
        code_tokens = tokenizer.tokenize(node_content)
        tokens = [tokenizer.cls_token] +  [tokenizer.sep_token] + code_tokens + [tokenizer.eos_token]
        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
        # tokens =my_tokenizer(node_content)#对这行代码进行格式化，返回列表
        nrp = np.zeros(100)
        for token in tokens:#tokens列表 列表中存放每个单词
            try:
                embedding = wv.wv[token]#向量化
            except:
                embedding = np.zeros(100)
            nrp = np.add(nrp, embedding)#加到一起指代该结点的向量 1*100
        if len(tokens) > 0:
            fnrp = np.divide(nrp, len(tokens))#平均操作指代该结点的向量 1*100
        else:
            fnrp = nrp
        # fnrp = np.add(fnrp_cfg,fnrp_ast)
        #fnrp = np.concatenate((fnrp, one_hot), axis = 0)
        gInput.append(fnrp.tolist())
    return gInput
def word2vec1(nodes, index_map, graph,wv):
    gInput = list()
    all_nodes = set()
    for item in graph:
        s, _, e = item
        all_nodes.add(e)
        all_nodes.add(s)
    if len(all_nodes) != len(index_map):
        print("Process Error!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None
    for i in index_map:
        true_id = index_map[i]
        node = get_nodes_by_key(nodes, true_id)
        node_content = node['code'].strip()
        #node_type = node_type_map[node['type'].strip()] - 1
        #one_hot = np.zeros(len(node_type_map))
        #one_hot[node_type] = 1.0
        #tokens = nltk.word_tokenize(node_content)
        if node['isCFGNode']:
            embedding = sentence_embedding(node_content)
            fnrp =np.array(embedding)
            gInput.append(fnrp.tolist())
            continue
        tokens = my_tokenizer(node_content)
        nrp = np.zeros(100)
        for token in tokens:
            try:
                embedding = wv.wv[token]
            except:
                embedding = np.zeros(100)
            nrp = np.add(nrp, embedding)
        if len(tokens) > 0:
            fnrp = np.divide(nrp, len(tokens))
        else:
            fnrp = nrp
        #fnrp = np.concatenate((fnrp, one_hot), axis = 0)
        gInput.append(fnrp.tolist())
    return gInput
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='normalized csv files to process', default='./dataset/ffmpeg/csv/')  #nodes  edges
    parser.add_argument('--src', help='source c files to process', default='./dataset/ffmpeg/c_file/')
    parser.add_argument('--json_files', help = 'train and test and valid', default=['./dataset/ffmpeg/ffmpeg_data_split/train_raw_code.json',
    './dataset/ffmpeg/ffmpeg_data_split/test_raw_code.json', './dataset/ffmpeg/ffmpeg_data_split/val_raw_code.json'])
    parser.add_argument('--wv', default='./dataset/reveal/reveal_data')  # word2vc
    parser.add_argument('--output_dir', default='./dataset/reveal/cpg/test/')
    args = parser.parse_args()
    model = Word2Vec.load(args.wv)
    train_path, test_path, valid_path = args.json_files  #[0], args.json_files[1], args.json_files[2]
    # train_data = []
    # test_data = []
    # valid_data = []
    with open(train_path, 'r') as f:#train_raw_code.json
        train_data = json.load(f)#list
        print(len(train_data))
    with open(test_path, 'r') as f:#test_raw_code.json
        test_data = json.load(f)#list
        print(len(test_data))
    with open(valid_path, 'r') as f:#val_raw_code.json
        valid_data = json.load(f)#list
        print(len(valid_data))

    data = [train_data, test_data, valid_data]  # data是列表,train_data，test_data,valid_data等也为列表,train_data[i]字典
    output_dir = args.output_dir  # ./outputs/devign_cpg_c0
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("*"*100)

    train_output_path = open(os.path.join(output_dir, 'devign-train-v0.json'), 'w')
    test_output_path = open(os.path.join(output_dir, 'devign-test-v0.json'), 'w')
    valid_output_path = open(os.path.join(output_dir, 'devign-valid-v0.json'), 'w')
    output_files = [train_output_path, test_output_path, valid_output_path]

    train_num_path = open(os.path.join(output_dir, 'devign-train-num-v0.json'), 'w')
    test_num_path = open(os.path.join(output_dir, 'devign-test-num-v0.json'), 'w')
    valid_num_path = open(os.path.join(output_dir, 'devign-valid-num-v0.json'), 'w')
    num_files = [train_num_path, test_num_path, valid_num_path]

    train_file = open(os.path.join(output_dir, 'devign-train-file.json'), 'w')
    test_file = open(os.path.join(output_dir, 'devign-test-file.json'), 'w')
    valid_file = open(os.path.join(output_dir, 'devign-valid-file.json'), 'w')
    file_names = [train_file, test_file, valid_file]

    bad_file = []
    bad_file_path = open(os.path.join(output_dir, 'bad_file.json'), 'w')

    for i in range(0, 3): #i=0,1,2
        final_data = []
        final_num = []
        files = []
        num = 0
        for _, entry in enumerate(tqdm(data[i])):#data[i]:train_data/test_data/valid_data:列表  遍历列表
            #data[i]指代一个源函数
            # {
            #     "file_path": "14015_1.c",
            #     "label": 1,
            #     "code": "static int ape_read_headerio_rl32(pb)......
            # },
            file_name = entry['file_path']
            nodes_path = os.path.join(args.csv, file_name,'tmp',file_name,'nodes.csv')
            edges_path = os.path.join(args.csv, file_name,'tmp',file_name,'edges.csv')
            label = int(entry['label'])
            if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
                continue
            #读这个源函数通过joern生成的csv
            nodes = read_csv(nodes_path)#返回一个列表，列表的内容是字典，是该csv文件中除了第一行，每一行内容对应的字典，键是表头，值是该行表头对应的值
            edges = read_csv(edges_path)#nodes,edges都对应一个c文件
            #nodes/edges 都是列表
            index_map, graph, nodes_num, edges_num = graphGeneration(nodes, edges, edgeType_reduced, ver_edge_type_reduced)
            # return index_map_ver, all_edges_new 类型为数字, len(index_map_ver), edges_num
            if edges_num == True:
                bad_file.append(file_name)
            if index_map is None or graph is None or nodes_num is None or edges_num is None:
                continue
            gInput = word2vec(nodes, index_map, graph, model)#ginput列表 列表元素为每个节点的向量
            if gInput is None:
                continue
            # if check(index_map, graph, gInput) != True:
            #     print("check error!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #     continue
            data_point = {
                'node_features': gInput,#ginput形状：节点数量*100
                'graph': graph,#src type dest
                'targets': [[label]]
            }
            num_point = {
                'file_name' : file_name,
                'nodes_num' : nodes_num,
                'edges_num' : edges_num
            }
            num += 1
            files.append(file_name)
            final_data.append(data_point)
            final_num.append(num_point)
        print(num)
        json.dump(final_data, output_files[i])
        json.dump(final_num, num_files[i])
        json.dump(files, file_names[i])
        output_files[i].close()
        num_files[i].close()
        file_names[i].close()
    json.dump(bad_file, bad_file_path)
    bad_file_path.close()

if __name__ == '__main__':
    main()