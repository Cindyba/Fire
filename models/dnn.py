import torch.nn.functional as F
import torch.nn as nn
import torch

class Dnn_Net(nn.Module):
    def __init__(self, in_dim, n_hidden):
        super(Dnn_Net, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_hidden)
        self.layer4= nn.Linear(n_hidden, 1)
        self.norm = nn.BatchNorm1d(1)
        self.activation = nn.ELU()
        self.norm_layer = torch.nn.LayerNorm(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.layer4(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.norm_layer(x)
        return x

class Dnn_Embedding(nn.Module):
    def __init__(self, in_dim, n_hidden):
        super(Dnn_Embedding, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_hidden)
        self.norm = nn.BatchNorm1d(n_hidden)
        self.activation = nn.ELU()
        self.norm_layer = torch.nn.LayerNorm(n_hidden)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.norm_layer(x)
        return x