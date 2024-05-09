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
    def __init__(self, input_dim, output_dim, population_size=300, to_mate=150, to_mutate=50, hidden_units=10, hidden_layers=1,
                 dropout_percent=0, step_size=.1, activation=nn.ReLU(), output_activation=nn.Softmax(dim=-1)):

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
        self.population_size = population_size
        self.to_mate = to_mate
        self.to_mutate = to_mutate
        self.population = None

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

    def generate_initial_population(self, size, model):
        initial_population = []
        with torch.no_grad():
            for _ in range(size):
                new_model = deepcopy(model)
                for new_param, param in zip(new_model.parameters(), model.parameters()):
                    if len(param.shape) > 1:  # using weight matrices
                        new_param.data = param.data + torch.randn_like(param) * 0.1
                initial_population.append(new_model)
        return initial_population

    def evaluate(self, individual, criterion):
        # criterion = torch.nn.CrossEntropyLoss()
        # criterion = net.criterion
        individual.eval()
        with torch.no_grad():
            outputs = individual(self.data)
            loss = criterion(outputs, self.targets)
        return -loss.item()

    def mate(self, parent1, parent2):
        child = deepcopy(parent1)
        for child_param, parent1_param, parent2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            if len(child_param.shape) > 1:  # mate weights
                mask = torch.bernoulli(torch.full_like(parent1_param.data, 0.5))
                child_param.data = mask * parent1_param.data + (1 - mask) * parent2_param.data
        return child

    def mutate(self, individual):
        mutation_strength = 0.1
        for param in individual.parameters():
            if len(param.shape) > 1:  # mutate weights
                if np.random.rand() < mutation_strength:
                    noise = torch.randn_like(param) * 0.1
                    param.data += noise

    def run_ga_single_step(self, net, X_train, y_train, **fit_params):
        # copy weights
        #net.save_params(f_params='sa_model_params.pt', f_optimizer='sa_optimizer_params.pt')

        # calc old loss
        y_pred = net.infer(X_train, **fit_params)
        loss = net.get_loss(y_pred, y_train, X_train, training=False)

        model = net.module_
        self.data = X_train
        self.targets = y_train

        if self.population is None:
            self.population = self.generate_initial_population(self.population_size, model)

        self.values = np.array([self.evaluate(individual, net.criterion) for individual in self.population])

        # Calculate probabilities for selection based on fitness
        fitness = np.array(self.values)
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
            new_values[i] = self.values[index]

        # Mutation phase
        for i in range(self.to_mutate):
            index = np.random.randint(self.population_size)
            self.mutate(new_population[index])
            new_values[index] = -1  # Mark for re-evaluation

        # Re-evaluate new population
        for i in range(self.population_size):
            if new_values[i] == -1:
                new_values[i] = self.evaluate(new_population[i], net.criterion)

        self.population = new_population
        self.values = new_values
        best_fitness = -np.min(self.values)
        best_fitness_index = np.argmin(self.values)

        #print(f"Best Fitness {best_fitness}")
        #print(f"loss {loss}")

        old_model = net.module_
        net.module_ = deepcopy(self.population[best_fitness_index])
        # calc new loss
        new_y_pred = net.infer(X_train, **fit_params)
        new_loss = net.get_loss(new_y_pred, y_train, X_train, training=False)

        if new_loss > loss:
            net.module_ = deepcopy(old_model)
            new_y_pred = y_pred
            new_loss = loss
            #print("swap models", best_fitness_index, best_fitness, self.evaluate(self.population[best_fitness_index]))
            #todo: optimize
            #todo: undo unecessary deepcopy above - only copy when new model is better
            #todo: consider option to regenerate init population during training

            #net.module_ = deepcopy(self.population[best_fitness_index])
            #print(f"old loss {loss}")
            # calc new loss
            #y_pred = net.infer(X_train, **fit_params)
            #loss = net.get_loss(y_pred, y_train, X_train, training=False)
            #print("swap models, best fitness", best_fitness)
            #print(f"new loss {loss}")

        return new_loss, new_y_pred

    @staticmethod
    def register_ga_training_step():
        """
        train_step_single override - add GA training step and disable backprop
        """
        @add_to(NeuralNet)
        def train_step_single(self, batch, **fit_params):
            self._set_training(False)
            Xi, yi = unpack_data(batch)
            # disable backprop and run custom training step
            # loss.backward()
            loss, y_pred = self.module_.run_ga_single_step(self, Xi, yi, **fit_params)
            return {
                'loss': loss,
                'y_pred': y_pred,
            }
