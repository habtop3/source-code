import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""

# self.MPL_layer = MLPReadout(output_dim, 2)
class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        # print(self.FC_layers)
        self.L = L

    def forward(self, x):
        y = x#64*100
        for l in range(self.L):#（0，1）
            y = self.FC_layers[l](y)
            y = F.relu(y)
            # print(y.shape)
        y = self.FC_layers[self.L](y)
        return y