import argparse
import yaml
import logging
import os
import pickle
import sys
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from data_loader.dataset import DataSet
from modules.model import DevignModel
from trainer import train
from utils import tally_param, debug, set_logger
import math
from torch.optim.optimizer import Optimizer
import torch.optim
import warnings

warnings.filterwarnings('ignore')
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
os.chdir(sys.path[0])
print(torch.cuda.is_available())

yaml_path = "./config/ffmpeg.yami"

def read_yaml(yaml_path):
    # 使用open()函数读取config.yaml文件
    yaml_file = open(yaml_path, "r", encoding="utf-8")
    # 读取文件中的内容
    file_data = yaml_file.read()
    yaml_file.close()
    # 加载数据流，返回字典类型数据
    y = yaml.load(file_data, Loader=yaml.FullLoader)
    # 下面就可以使用字典访问配置文件中的数据了
    return y
class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-6,
                 weight_decay=5e-4):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)

        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        beta2_t = None
        ratio = None
        N_sma_max = None
        N_sma = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                if beta2_t is None:
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    beta1_t = 1 - beta1 ** state['step']
                    if N_sma >= 5:
                        ratio = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / beta1_t

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:                    
                    step_size = group['lr'] * ratio
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group['lr'] / beta1_t
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

if __name__ == '__main__':
    config = read_yaml(yaml_path)


    torch.manual_seed(10)
    np.random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--dataset', type=str, help='Name of the dataset for experiment.', default='reveal')
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser', default='../data_processing/dataset/reveal/outputs/reveal_cpg/')
    parser.add_argument('--log_dir', default='orign_ff.log', type=str)
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='targets')
    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=64)
    parser.add_argument('--k', type=int, help='knn', default=1)
    args = parser.parse_args()

    dataset = config['dataset']
    model_dir = config['model_dir']
    log_dir = config['log_dir']
    feature_size = config['feature_size']
    graph_embed_size = config['graph_embed_size']
    input_dir = config['input_dir']
    if not os.path.exists(model_dir): #判断是否存在
        os.makedirs(model_dir)
    set_logger(log_dir)
    logging.info('Check up feature_size: %d', feature_size)
    if feature_size > graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        logging.info('Warning!!! Graph Embed dimension should be at least equal to the feature dimension')
        graph_embed_size = feature_size

    input_dir = config['input_dir']#../output/devign_cpg_c0/
    processed_data_path = config['processed_data_path'] #processed_data_path=dataset/Devign/devign_cpg_c2_2/devign.bin
    logging.info('#' * 100)

    if True and os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        logging.info('Reading already processed data from %s!' % processed_data_path)
    else:
        logging.info('Loading the dataset from %s' % input_dir)
        dataset = DataSet(train_src = config['train_src'],
                          valid_src = config['valid_src'],
                          test_src = config['test_src'],
                          batch_size=config['batch_size'],
                          n_ident=config['node_tag'],
                          g_ident=config['graph_tag'],
                          l_ident=config['label_tag'],
                          )
        file = open(processed_data_path, 'wb')#processed_data_path:../output/devign_cpg_c0/devign.bin
        pickle.dump(dataset, file)
        file.close()
    logging.info('train_dataset: %d; valid_dataset: %d; test_dataset: %d', len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    logging.info("train_batch: %d, valid_batch: %d, test_batch: %d", len(dataset.train_batches), len(dataset.valid_batches), len(dataset.test_batches))
    logging.info('#' * 100)
    logging.info('Check up model_type: ' + args.model_type)
    print(dataset.max_edge_type)  # 5
    print(config['num_steps'])  # 6
    print(dataset.edge_types)
    kk
    if args.model_type == 'ggnn':
        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    else:
        model = DevignModel(input_dim=dataset.feature_size, #100
                            output_dim=100,
                            num_steps=args.num_steps, #6
                            max_edge_types=dataset.max_edge_type,#5
                            dropout=0.5,
                            nfeat =  100,
                            nhid1 = 200,
                            nhid2 = 100,
                            nclass= 2
        )

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    logging.info('Total Parameters : %d' % tally_param(model))
    logging.info('#' * 100)

    model.cuda()#把模型移到gpu上

    loss_function = CrossEntropyLoss(weight=torch.from_numpy(np.array([1,1.2])).float(),reduction='sum')
    loss_function.cuda()

    LR = 1e-5
    optim = RAdam(model.parameters(),lr=LR,weight_decay=1e-6)
    dataset_name = args.dataset

    train(model=model, dataset=dataset, epoches=100, dev_every=len(dataset.train_batches),
          loss_function=loss_function, optimizer=optim,
          save_path=model_dir + '/'+f'{args.dataset}Model',datasetname=dataset_name,k=args.k, max_patience=100, log_every=5)