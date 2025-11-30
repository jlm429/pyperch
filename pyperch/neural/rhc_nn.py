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
                 output_activation=nn.Softmax(dim=-1), random_seed=None, trainable_layers=None, freeze_seed=None):
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

        trainable_layers {int or None}:
            Number of layers to keep unfrozen (counting from the end excl input layer). If None, all layers are trainable.
            If specified, only the last trainable_layers will be trainable, others will be frozen.

        freeze_seed {int or None}:
            Random seed for reproducible freezing. If None, uses random_seed.
    
        trainable_params_counter {int}:
            Counter of trainable parameters.

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
        self.random_seed = random_seed
        self.trainable_layers = trainable_layers
        self.freeze_seed = freeze_seed if freeze_seed is not None else random_seed
        
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Create layers based on layer_sizes
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1], device=self.device))
        
        # 0 if there's freezing required to be calc'd in _apply_freezing else will be sum of all params since all layers are trainable
        self.trainable_params_counter = 0 if trainable_layers is not None else sum(param.numel() for param in self.layers.parameters())
        
        # Apply freezing if specified
        self._apply_freezing()

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

    def _apply_freezing(self):
        """
        Apply freezing to layers based on trainable_layers parameter.
        Freezes all layers except the last trainable_layers layers.
        """
        if self.trainable_layers is None:
            print("GA: No freezing applied - all layers are trainable")
            return
        
        if self.trainable_layers <= 0:
            raise ValueError(f"GA: trainable_layers must be > 0, got {self.trainable_layers}. Use trainable_layers=None to train all layers.")
            
        if self.trainable_layers >= len(self.layers):
            print("GA: Warning - trainable_layers >= total layers, all layers will be trainable")
            return
        
        # randoms seed
        if self.freeze_seed is not None:
            torch.manual_seed(self.freeze_seed)
            np.random.seed(self.freeze_seed)
        
        # calc which layers to freeze and freeze em
        layers_to_freeze = len(self.layers) - self.trainable_layers 
        print(f"GA: Freezing first {layers_to_freeze} layers, keeping last {self.trainable_layers} layers trainable")
        for i in range(layers_to_freeze):
            for param in self.layers[i].parameters():
                self.trainable_params_counter += param.numel()
                param.requires_grad = False
            print(f"GA: Layer {i} frozen (size: {self.layer_sizes[i]} -> {self.layer_sizes[i+1]})")
        
        # reset random seed to original if different from freeze_seed
        if self.random_seed is not None and self.random_seed != self.freeze_seed:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

    def _is_layer_trainable(self, layer: nn.Module) -> bool:
            """Checks if all parameters in a PyTorch layer are trainable (requires_grad=True)."""
            for param in layer.parameters():
                # assumption:
                # if we find even one parameter that requires a gradient, the layer is trainable.
                if param.requires_grad:
                    return True
            return False
        
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

        # select random layer (only from trainable layers, for unfrozen case will list all anyway)
        trainable_layers = [i for i, layer in enumerate(net.module_.layers) if any(param.requires_grad for param in layer.parameters())]
        if not trainable_layers:
            print("RHC: Warning - no trainable layers available for optimization")
            return loss, y_pred
        
        layer_idx = np.random.choice(trainable_layers)
        layer = net.module_.layers[layer_idx]

        # omitted: this is redundant check since we're alr selecting from a list of trainable layers
        # if self._is_layer_trainable(layer): 
        input_dim = np.random.randint(0, layer.weight.shape[0])
        output_dim = np.random.randint(0, layer.weight.shape[1])
        neighbor = self.step_size * np.random.choice([-1, 1])

        with torch.no_grad():
            layer.weight[input_dim][output_dim] = neighbor + layer.weight[input_dim][output_dim].data

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
