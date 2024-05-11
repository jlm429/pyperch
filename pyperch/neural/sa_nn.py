"""
Author: John Mansfield
BSD 3-Clause License

SAModule class: create a neural network model to be used with simulated annealing randomized optimization of weights.

Inspired by ABAGAIL - neural net simulated annealing implementation.

https://github.com/pushkar/ABAGAIL/blob/master/src/func/nn/OptNetworkBuilder.java
"""

import numpy as np
import torch
from torch import nn
from skorch import NeuralNet
from pyperch.utils.decorators import add_to
from skorch.dataset import unpack_data
import copy
import math


class SAModule(nn.Module):
    def __init__(self, input_dim, output_dim, t=10000, cooling=.95, hidden_units=10, hidden_layers=1,
                 dropout_percent=0, step_size=.1, activation=nn.ReLU(), output_activation=nn.Softmax(dim=-1)):
        """

        Initialize the neural network.

        PARAMETERS:

        input_dim {int}:
            Number of features/dimension of the input.  Must be greater than 0.

        output_dim {int}:
            Number of classes/output dimension of the model. Must be greater than 0.

        t {int}:
            SA temperature.

        cooling {float}:
            Cooling rate.

        hidden_units {int}:
            Number of hidden units.

        hidden_layers {int}:
            Number of hidden layers.

        dropout_percent {float}:
            Probability of an element to be zeroed.

        step_size {float}:
            Step size for hill climbing.

        activation {torch.nn.modules.activation}:
            Activation function.

        output_activation {torch.nn.modules.activation}:
            Output activation.

        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.dropout = nn.Dropout(dropout_percent)
        self.step_size = step_size
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.t = t
        self.cooling = cooling

        # input layer
        self.layers.append(nn.Linear(self.input_dim, self.hidden_units, device=self.device))
        # hidden layers
        for layer in range(self.hidden_layers):
            self.layers.append(nn.Linear(self.hidden_units, self.hidden_units, device=self.device))
        # output layer
        self.layers.append(nn.Linear(self.hidden_units, self.output_dim, device=self.device))

    def forward(self, X, **kwargs):
        """
        Recipe for the forward pass.

        PARAMETERS:

        X {torch.tensor}:
            NN input data. Shape (batch_size, input_dim).

        RETURNS:

        X {torch.tensor}:
            NN output data. Shape (batch_size, output_dim).
        """
        X = self.activation(self.layers[0](X))
        X = self.dropout(X)
        for i in range(self.hidden_layers):
            X = self.activation(self.layers[i+1](X))
        X = self.output_activation(self.layers[self.hidden_layers+1](X))
        return X

    def run_sa_single_step(self, net, X_train, y_train, **fit_params):
        """
        SA training step

        PARAMETERS:

        net {skorch.classifier.NeuralNetClassifier}:
            Skorch NeuralNetClassifier.

        X_train {torch.tensor}:
            Training data.

        y_train {torch.tensor}:
            Training labels.

        RETURNS:

        loss {torch.tensor}:
            Single step loss.

        y_pred {torch.tensor}:
            Predicted labels.
        """
        # copy weights
        net.save_params(f_params='sa_model_params.pt', f_optimizer='sa_optimizer_params.pt')

        # calc old loss
        y_pred = net.infer(X_train, **fit_params)
        loss = net.get_loss(y_pred, y_train, X_train, training=False)

        # select random layer
        layer = np.random.randint(0, len(self.layers))-1
        input_dim = np.random.randint(0, net.module_.layers[layer].weight.shape[0])
        output_dim = np.random.randint(0, net.module_.layers[layer].weight.shape[1])
        neighbor = self.step_size * np.random.choice([-1, 1])

        with torch.no_grad():
            net.module_.layers[layer].weight[input_dim][output_dim] = neighbor + \
                net.module_.layers[layer].weight[input_dim][output_dim].data

        # Evaluate new loss
        new_y_pred = net.infer(X_train, **fit_params)
        new_loss = net.get_loss(new_y_pred, y_train, X_train, training=False)

        # Calculate the change in objective function value
        delta = new_loss - loss

        # If the new solution is better or with probability e^(delta/temperature), accept it
        if new_loss > loss and np.random.rand() >= math.exp(-delta / self.t):
            net.load_params(f_params='sa_model_params.pt', f_optimizer='sa_optimizer_params.pt')
            new_y_pred = y_pred
            new_loss = loss

        self.t = self.cooling * self.t
        return new_loss, new_y_pred

    @staticmethod
    def register_sa_training_step():
        """
        train_step_single override - add SA training step and disable backprop
        """
        @add_to(NeuralNet)
        def train_step_single(self, batch, **fit_params):
            self._set_training(False)
            Xi, yi = unpack_data(batch)
            # disable backprop and run custom training step
            # loss.backward()
            loss, y_pred = self.module_.run_sa_single_step(self, Xi, yi, **fit_params)
            return {
                'loss': loss,
                'y_pred': y_pred,
            }
