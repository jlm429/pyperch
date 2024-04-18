"""
Author: John Mansfield
BSD 3-Clause License

Backprop class: create a backprop neural network model.
"""

import torch
from torch import nn


class BackpropModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=10, hidden_layers=1,
                 dropout_percent=0, activation=nn.ReLU(), output_activation=nn.Softmax(dim=-1)):
        """

        Initialize the neural network.

        PARAMETERS:

        input_dim {int}:
            Number of features/dimension of the input.  Must be greater than 0.

        output_dim {int}:
            Number of classes/output dimension of the model. Must be greater than 0.

        hidden_units {int}:
            Number of hidden units.

        hidden_layers {int}:
            Number of hidden layers.

        dropout_percent {float}:
            Probability of an element to be zeroed.

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
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
