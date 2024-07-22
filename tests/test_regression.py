import unittest
import numpy as np
import torch
from sklearn.datasets import make_regression
from torch import nn, optim
from skorch import NeuralNetRegressor
from skorch.callbacks import EpochScoring
from pyperch.neural.backprop_nn import BackpropModule
from pyperch.neural.rhc_nn import RHCModule
from pyperch.neural.sa_nn import SAModule
from pyperch.neural.ga_nn import GAModule


class TestRegressionNetworks(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.X, self.y = make_regression(n_samples=1000, n_features=12, n_informative=10, noise=.2, random_state=seed)
        self.X = self.X.astype(np.float32)
        self.y = self.y.reshape(-1, 1).astype(np.float32)

    def test_dataset_creation(self):
        self.assertEqual(self.X.shape, (1000, 12))
        self.assertEqual(self.y.shape, (1000, 1))
        self.assertTrue(np.issubdtype(self.X.dtype, np.float32))
        self.assertTrue(np.issubdtype(self.y.dtype, np.float32))

    def test_backprop_neural_network_initialization_and_fitting(self):
        self.net = NeuralNetRegressor(
            module=BackpropModule,
            module__layer_sizes=(12, 32, 1),
            criterion=nn.MSELoss(),
            module__activation=nn.LeakyReLU(),
            module__output_activation=lambda x: x,
            max_epochs=50,
            verbose=0,
            callbacks=[EpochScoring(scoring='r2', name='train_r2', on_train=True),
                       EpochScoring(scoring='r2', name='validation_r2', on_train=False)],
            lr=.0001,
            iterator_train__shuffle=True,
        )
        self.net.fit(self.X, self.y)
        self.assertTrue(hasattr(self.net, 'history_'))
        self.assertEqual(len(self.net.history), 50)

    def test_rhc_neural_network_initialization_and_fitting(self):
        self.net = NeuralNetRegressor(
            module=RHCModule,
            module__layer_sizes=(12, 32, 1),
            module__step_size=.05,
            max_epochs=50,
            verbose=0,
            module__output_activation=lambda x: x,
            criterion=nn.MSELoss(),
            callbacks=[EpochScoring(scoring='r2', name='train_r2', on_train=True),
                       EpochScoring(scoring='r2', name='validation_r2', on_train=False)],
            iterator_train__shuffle=True,
        )
        self.net.fit(self.X, self.y)
        self.assertTrue(hasattr(self.net, 'history_'))
        self.assertEqual(len(self.net.history), 50)

    def test_sa_neural_network_initialization_and_fitting(self):
        self.net = NeuralNetRegressor(
            module=SAModule,
            module__layer_sizes=(12, 32, 1),
            module__step_size=.05,
            max_epochs=50,
            verbose=0,
            # module__activation=nn.LeakyReLU(),
            criterion=nn.MSELoss(),
            module__output_activation=lambda x: x,
            callbacks=[EpochScoring(scoring='r2', name='train_r2', on_train=True),
                       EpochScoring(scoring='r2', name='validation_r2', on_train=False)],
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
        )
        self.net.fit(self.X, self.y)
        self.assertTrue(hasattr(self.net, 'history_'))
        self.assertEqual(len(self.net.history), 50)

    def test_ga_neural_network_initialization_and_fitting(self):
        self.net = NeuralNetRegressor(
            module=GAModule,
            module__layer_sizes=(12, 32, 1),
            module__population_size=300,
            module__to_mate=30,
            module__to_mutate=10,
            max_epochs=2,
            verbose=0,
            criterion=nn.MSELoss(),
            module__output_activation=lambda x: x,
            callbacks=[EpochScoring(scoring='r2', name='train_r2', on_train=True),
                       EpochScoring(scoring='r2', name='validation_r2', on_train=False)],
        )
        self.net.fit(self.X, self.y)
        self.assertTrue(hasattr(self.net, 'history_'))
        self.assertEqual(len(self.net.history), 2)


if __name__ == '__main__':
    unittest.main()