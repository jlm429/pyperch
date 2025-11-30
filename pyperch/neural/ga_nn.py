"""
Author: John Mansfield
BSD 3-Clause License

GAModule class: create a neural network model to be used with genetic algorithm randomized optimization of weights.

Inspired by ABAGAIL - neural net genetic algorithm implementation.

https://github.com/pushkar/ABAGAIL/blob/master/src/func/nn/OptNetworkBuilder.java
"""
import numpy as np
import torch
from torch import nn
from skorch import NeuralNet
from pyperch.utils.decorators import add_to
from skorch.dataset import unpack_data
from copy import deepcopy


class GAModule(nn.Module):
    def __init__(self, layer_sizes, population_size=300, to_mate=150, to_mutate=50, dropout_percent=0,
                 step_size=.1, activation=nn.ReLU(), output_activation=nn.Softmax(dim=-1), random_seed=None,
                 trainable_layers=None, freeze_seed=None):
        """

        Initialize the neural network.

        PARAMETERS:

        layer_sizes {array-like}:
            Sizes of all layers including input, hidden, and output layers. Must be a tuple or list of integers.

        population_size {int}:
            GA population size.  Must be greater than 0.

        to_mate {int}:
            GA size of population to mate each time step.

        to_mutate {int}:
            GA size of population to mutate each time step.

        dropout_percent {float}:
            Probability of an element to be zeroed.

        step_size {float}:
            Step size for mutation strength

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
        GAModule.register_ga_training_step()
        self.layer_sizes = layer_sizes
        self.dropout = nn.Dropout(dropout_percent)
        self.step_size = step_size
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.population_size = population_size
        self.to_mate = to_mate
        self.to_mutate = to_mutate
        self.population = None
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

    def generate_initial_population(self, size, model):
        """
        Generates an initial population of neural network models by introducing slight variations to
        the given model's weights. Only modifies trainable parameters.

        Parameters:

        size {int}:
            The size of the population to generate.

        model {torch.nn.Module}:
            The neural network model to be used to create the initial population.

        Returns:
            A list of neural network models that form the initial population.
        """
        initial_population = []
        with torch.no_grad():
            for _ in range(size):
                new_model = deepcopy(model)
                for new_param, param in zip(new_model.parameters(), model.parameters()):
                    # Only modify trainable parameters
                    if param.requires_grad and len(param.shape) > 1:  # using weight matrices
                        new_param.data = param.data + torch.randn_like(param) * 0.1
                initial_population.append(new_model)
        return initial_population

    def evaluate(self, individual, criterion, data, targets):
        """
        Evaluates a given neural network model's performance by calculating its fitness based on how well it predicts the target values from the given data.

        Parameters:

        individual {torch.nn.Module}:
            The neural network model to evaluate.

        Returns:
            The fitness score (negative loss) of the model.
        """
        individual.eval()
        with torch.no_grad():
            outputs = individual(data)
            loss = criterion(outputs, targets)
        return -loss.item()

    def mate(self, parent1, parent2):
        """
        Combines weights of two parent models using uniform crossover. Each gene (nn weight) from the child
        model is selected randomly from one of the two parents with equal probability.
        Only operates on trainable parameters.

        Parameters:

        parent1 {torch.nn.Module}:
            The first parent model.

        parent2 {torch.nn.Module}:
            The second parent model.

        Returns:
            A new neural network model that is a combination of weights from both parents.
        """
        child = deepcopy(parent1)
        for child_param, parent1_param, parent2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            # Only mate trainable parameters
            if parent1_param.requires_grad and len(child_param.shape) > 1:  # mate weights
                mask = torch.bernoulli(torch.full_like(parent1_param.data, 0.5))
                child_param.data = mask * parent1_param.data + (1 - mask) * parent2_param.data
        return child

    def mutate(self, individual):
        """
        Introduces random mutations to the neural net weights.
        Only mutates trainable parameters.

        Parameters:

        individual {torch.nn.Module}:
            The model to mutate.
        """
        mutation_strength = self.step_size
        for param in individual.parameters():
            # Only mutate trainable parameters
            if param.requires_grad and len(param.shape) > 1:  # mutate weights
                if np.random.rand() < mutation_strength:
                    noise = torch.randn_like(param) * 0.1
                    param.data += noise

    def run_ga_single_step(self, net, X_train, y_train, **fit_params):
        """
        GA training step

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
        # calc old loss
        y_pred = net.infer(X_train, **fit_params)
        loss = net.get_loss(y_pred, y_train, X_train, training=False)

        model = net.module_
        data = X_train
        targets = y_train

        if self.population is None:
            self.population = self.generate_initial_population(self.population_size, model)

        values = np.array([self.evaluate(individual, net.criterion, data, targets) for individual in self.population])

        # Calculate probabilities for selection based on fitness
        fitness = np.array(values)
        probabilities = fitness - fitness.min()
        if probabilities.sum() > 0:
            probabilities /= probabilities.sum()
        else:
            probabilities = np.ones(self.population_size) / self.population_size

        new_population = []
        new_values = np.zeros(self.population_size)

        # Mate phase
        for i in range(self.to_mate):
            parents = np.random.choice(self.population_size, 2, p=probabilities)
            child = self.mate(self.population[parents[0]], self.population[parents[1]])
            new_population.append(child)
            new_values[i] = -1  # Mark for re-evaluation

        # Elite selection
        for i in range(self.to_mate, self.population_size):
            index = np.random.choice(self.population_size, p=probabilities)
            new_population.append(deepcopy(self.population[index]))
            new_values[i] = values[index]

        # Mutation phase
        for i in range(self.to_mutate):
            index = np.random.randint(self.population_size)
            self.mutate(new_population[index])
            new_values[index] = -1  # Mark for re-evaluation

        # Re-evaluate new population
        for i in range(self.population_size):
            if new_values[i] == -1:
                new_values[i] = self.evaluate(new_population[i], net.criterion, data, targets)

        self.population = new_population
        values = new_values
        best_fitness_index = np.argmin(values)

        old_model = net.module_
        net.module_ = deepcopy(self.population[best_fitness_index])

        # calc new loss
        new_y_pred = net.infer(X_train, **fit_params)
        new_loss = net.get_loss(new_y_pred, y_train, X_train, training=False)

        # Revert to old weights if new loss is higher
        if new_loss > loss:
            net.module_ = deepcopy(old_model)
            new_y_pred = y_pred
            new_loss = loss

        return new_loss, new_y_pred

    @staticmethod
    def register_ga_training_step():
        """
        train_step_single override - add GA training step and disable backprop
        """
        @add_to(NeuralNet)
        def train_step_single(self, batch, **fit_params):
            self._set_training(True)
            Xi, yi = unpack_data(batch)
            # disable backprop and run custom training step
            # loss.backward()
            loss, y_pred = self.module_.run_ga_single_step(self, Xi, yi, **fit_params)
            return {
                'loss': loss,
                'y_pred': y_pred,
            }
