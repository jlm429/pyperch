import torch
from torch import nn


ACTS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class SimpleMLP(nn.Module):
    """
    A configurable feed-forward MLP.

    Parameters:
        input_dim (int): Number of input features.
        hidden (list[int]): Hidden layer sizes.
        output_dim (int): Output units (2 for binary, N for multiclass, 1 for regression).
        activation (str): Activation function for hidden layers.
    """

    def __init__(self, input_dim, hidden, output_dim, activation="relu"):
        super().__init__()

        if activation not in ACTS:
            raise ValueError(f"Unknown activation '{activation}'")

        act = ACTS[activation]

        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act())  # initial activation
            prev = h

        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    # ---------------------------------------------------------
    # allow Trainer to override activation from ModelConfig
    # ---------------------------------------------------------
    def set_activation(self, activation: str):
        """Replace activation layers with the specified activation."""
        if activation not in ACTS:
            raise ValueError(f"Unknown activation '{activation}'")

        act_cls = ACTS[activation]

        new_layers = []
        for layer in self.net:
            # Replace any activation layer
            if isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid)):
                new_layers.append(act_cls())
            else:
                new_layers.append(layer)

        self.net = nn.Sequential(*new_layers)

    def forward(self, x):
        return self.net(x)
