import argparse
import yaml
import logging
import os
import pickle
import sys
import numpy as np
from AMPLE_code.data_loader.dataset import DataSet
from test.utils import debug
import torch.optim
import warnings
warnings.filterwarnings('ignore')
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
os.chdir(sys.path[0])
print(torch.cuda.is_available())

# LR = 1e-4
#     optim = RAdam(model.parameters(),lr=LR,weight_decay=1e-6)



yaml_path = "./config/datasetconfig/ffmpeg/ffmpeg.yami"

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

if __name__ == '__main__':
    config = read_yaml(yaml_path)
    torch.manual_seed(10)
    np.random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, help='knn', default=1)
    args = parser.parse_args()

    dataset = config['dataset']
    model_dir = config['model_dir']
    feature_size = config['feature_size']
    graph_embed_size = config['graph_embed_size']

    if not os.path.exists(model_dir): #判断是否存在
        os.makedirs(model_dir)

    debug('Check up feature_size: %d', feature_size)
    if feature_size > graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        debug('Warning!!! Graph Embed dimension should be at least equal to the feature dimension')
        graph_embed_size = feature_size

    input_dir = config['input_dir']#../output/devign_cpg_c0/
    processed_data_path = config['processed_data_path'] #processed_data_path=dataset/Devign/devign_cpg_c2_2/devign.bin
    debug('#' * 100)

    if True and os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        debug('Reading already processed data from %s!' % processed_data_path)
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
    debug('train_dataset: %d; valid_dataset: %d; test_dataset: %d', len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    debug("train_batch: %d, valid_batch: %d, test_batch: %d", len(dataset.train_batches), len(dataset.valid_batches), len(dataset.test_batches))
    debug('#' * 100)
    debug(dataset.max_edge_type)  # 5
    debug(config['num_steps'])  # 6
    debug(dataset.edge_types)