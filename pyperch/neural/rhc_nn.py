import numpy as np
from sklearn.datasets import make_classification
import torch
from torch import nn
from skorch import NeuralNetClassifier
from skorch import NeuralNet
from pyperch.utils.decorators import add_to
from skorch.dataset import unpack_data
import copy


class RHCModule(nn.Module):
    def __init__(self, input_dim=20, output_dim=2, num_units=10, nonlin=nn.ReLU()):
        super().__init__()

        #any input params can automagically be grid searched
        #TODO: add RHC specific params
        self.dense0 = nn.Linear(input_dim, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X

    #warning: this is only a POC for replacing the backprop training step
    def run_rhc_single_step(self, net, old_loss, X_train, y_train, **fit_params):
        # TODO: Call RHC function and pass weight vector instead of randomly changing the weights
        for param in net.module_.parameters():
            param.data += torch.randn(param.data.size())

