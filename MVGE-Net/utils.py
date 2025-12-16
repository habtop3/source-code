import numpy as np

from data_loader import n_identifier, g_identifier, l_identifier
import inspect
from datetime import datetime
import logging
# self.n_ident, self.g_ident, self.l_ident= load_default_identifiers(n_ident, g_ident, l_ident)
# n_ident=args.node_tag  node_features
# g_ident=args.graph_tag  graph
# l_ident=args.label_tag target
def load_default_identifiers(n, g, l):
    if n is None:
        n = n_identifier
    if g is None:
        g = g_identifier
    if l is None:
        l = l_identifier
    return n, g, l

# self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=False)
def initialize_batch(entries, batch_size, shuffle=False):
    total = len(entries)
    print(str(total)+'k'*35)
    indices = np.arange(0, total , 1) #返回数组  0,1,2,3,4 indices[0:64]=[0,1,2....,63]
    if shuffle:
        np.random.shuffle(indices)
    batch_indices = []  #新建列表
    start = 0
    end = len(indices)
    curr = start
    while curr < end:
        c_end = curr + batch_size
        if c_end > end:
            c_end = end
        batch_indices.append(indices[curr:c_end])#indices[curr:c_end]对总的数组进行切片，返回新数组
        #batch_indices[[0....63],[64.....127].....[]]
        curr = c_end
    return batch_indices[::-1]#反转列表


def tally_param(model):
    total = 0
    for param in model.parameters():
        total += param.data.nelement()
    return total


def debug(*msg, sep='\t'):
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    print('[' + str(time) + '] File \"' + file_name + '\", line ' + str(ln) + '  ', end='\t')
    for m in msg:
        print(m, end=sep)
    print('')

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="w", encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        #logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
