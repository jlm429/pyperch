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
    def __init__(self, input_dim=20, output_dim=2, hidden_units=10, nonlin=nn.ReLU()):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units

        #any input params can automagically be grid searched
        #TODO: add RHC specific params/add ability to dynamically add hidden layers
        self.dense0 = nn.Linear(self.input_dim, self.hidden_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(self.hidden_units, self.hidden_units)
        self.output = nn.Linear(self.hidden_units, self.output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X

    def run_rhc_single_step(self, net, X_train, y_train, **fit_params):
        previous_model = copy.deepcopy(net.module_)

        #calc old loss
        y_pred = net.infer(X_train, **fit_params)
        old_loss = net.get_loss(y_pred, y_train, X_train, training=False)

        lr=.1
        #todo: randomly select tensor instead of hidden only
        #todo: optimize
        input_dim = np.random.randint(0, self.hidden_units)
        output_dim = np.random.randint(0, self.hidden_units)
        neighbor = lr * np.random.choice([-1, 1])

        with torch.no_grad():
            net.module_.dense1.weight[input_dim][output_dim] = neighbor + net.module_.dense1.weight[input_dim][output_dim].data

        # Evaluate the new loss
        y_pred = net.infer(X_train, **fit_params)
        new_loss = net.get_loss(y_pred, y_train, X_train, training=False)
        loss = new_loss

        # Revert to old weights if new loss is higher
        if new_loss > old_loss:
            net.module_ = copy.deepcopy(previous_model)
            loss = old_loss

        return loss, y_pred

    @staticmethod
    def register_rhc_training_step():
        @add_to(NeuralNet)
        def train_step_single(self, batch, **fit_params):
            # disable backprop and run custom training step
            self._set_training(False)
            Xi, yi = unpack_data(batch)
            # loss.backward()
            loss, y_pred = self.module_.run_rhc_single_step(self, Xi, yi, **fit_params)
            return {
                'loss': loss,
                'y_pred': y_pred,
            }