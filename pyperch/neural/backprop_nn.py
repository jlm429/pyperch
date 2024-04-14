import torch
from torch import nn

class BackpropModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=10, hidden_layers=1,
                 dropout_percent=0, activation=nn.ReLU(), output_activation=nn.Softmax(dim=-1)):
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
        X = self.activation(self.layers[0](X))
        X = self.dropout(X)
        for i in range(self.hidden_layers):
            X = self.activation(self.layers[i+1](X))
        X = self.output_activation(self.layers[self.hidden_layers+1](X))
        return X
