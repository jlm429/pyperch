"""
Author: John Mansfield
BSD 3-Clause License

Backprop class: create a backprop neural network model.
"""

import torch
from torch import nn
from pyperch.utils.decorators import add_to
from skorch.dataset import unpack_data
from skorch import NeuralNet


class BackpropModule(nn.Module):
    def __init__(self, layer_sizes, dropout_percent=0, activation=nn.ReLU(), output_activation=nn.Softmax(dim=-1), random_seed=None):
        """

        Initialize the neural network.

        PARAMETERS:

        layer_sizes {array-like}:
            Sizes of all layers including input, hidden, and output layers. Must be a tuple or list of integers.

        dropout_percent {float}:
            Probability of an element to be zeroed.

        activation {torch.nn.modules.activation}:
            Activation function.

        output_activation {torch.nn.modules.activation}:
            Output activation.

        """
        super().__init__()
        BackpropModule.register_backprop_training_step()
        self.layer_sizes = layer_sizes
        self.dropout = nn.Dropout(dropout_percent)
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if random_seed is not None:
            torch.manual_seed(random_seed)

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

    @staticmethod
    def register_backprop_training_step():
        """
        train_step_single override - revert to backprop
        """
        @add_to(NeuralNet)
        def train_step_single(self, batch, **fit_params):
            self._set_training(True)
            Xi, yi = unpack_data(batch)
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=True)
            loss.backward()
            return {
                'loss': loss,
                'y_pred': y_pred,
            }