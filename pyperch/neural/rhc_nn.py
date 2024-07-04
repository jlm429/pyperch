"""
Author: John Mansfield
BSD 3-Clause License

RHCModule class: create a neural network model to be used with randomized hill climbing optimization of weights.

Inspired by ABAGAIL - neural net RHC implementation.

https://github.com/pushkar/ABAGAIL/blob/master/src/func/nn/OptNetworkBuilder.java
"""

import numpy as np
import torch
from torch import nn
from skorch import NeuralNet
from pyperch.utils.decorators import add_to
from skorch.dataset import unpack_data
import copy


class RHCModule(nn.Module):
    def __init__(self, layer_sizes, dropout_percent=0, step_size=.1, activation=nn.ReLU(),
                 output_activation=nn.Softmax(dim=-1)):
        """

        Initialize the neural network.

        PARAMETERS:

        layer_sizes {array-like}:
            Sizes of all layers including input, hidden, and output layers. Must be a tuple or list of integers.

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
        RHCModule.register_rhc_training_step()
        self.layer_sizes = layer_sizes
        self.dropout = nn.Dropout(dropout_percent)
        self.step_size = step_size
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create layers based on layer_sizes
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1], device=self.device))

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
        for i in range(len(self.layers) - 1):
            X = self.activation(self.layers[i](X))
            X = self.dropout(X)
        X = self.output_activation(self.layers[-1](X))
        return X

    def run_rhc_single_step(self, net, X_train, y_train, **fit_params):
        """
        RHC training step

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
        # save weights
        # net.save_params(f_params='rhc_model_params.pt', f_optimizer='rhc_optimizer_params.pt')
        previous_model = copy.deepcopy(net.module_)

        # calc current loss
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

        # Revert to old weights if new loss is higher
        if new_loss > loss:
            #net.load_params(f_params='rhc_model_params.pt', f_optimizer='rhc_optimizer_params.pt')
            net.module_ = copy.deepcopy(previous_model)
            new_y_pred = y_pred
            new_loss = loss

        return new_loss, new_y_pred

    @staticmethod
    def register_rhc_training_step():
        """
        train_step_single override - add RHC training step and disable backprop
        """
        @add_to(NeuralNet)
        def train_step_single(self, batch, **fit_params):
            self._set_training(True)
            Xi, yi = unpack_data(batch)
            # disable backprop and run custom training step
            # loss.backward()
            loss, y_pred = self.module_.run_rhc_single_step(self, Xi, yi, **fit_params)
            return {
                'loss': loss,
                'y_pred': y_pred,
            }
