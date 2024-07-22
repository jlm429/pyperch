import unittest
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, GridSearchCV
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from pyperch.neural.backprop_nn import BackpropModule


class TestBackpropNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.X, self.y = make_classification(1000, 12, n_informative=10, random_state=seed)
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.int64)

        self.net = NeuralNetClassifier(
            module=BackpropModule,
            module__layer_sizes=(12, 32, 20, 2),
            module__dropout_percent=(.1),
            max_epochs=50,
            verbose=0,
            callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), ],
            lr=.05,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
        )

    def test_dataset_creation(self):
        self.assertEqual(self.X.shape, (1000, 12))
        self.assertEqual(self.y.shape, (1000,))
        self.assertTrue(np.issubdtype(self.X.dtype, np.float32))
        self.assertTrue(np.issubdtype(self.y.dtype, np.int64))

    def test_neural_network_initialization_and_fitting(self):
        self.net.fit(self.X, self.y)
        self.assertTrue(hasattr(self.net, 'history_'))
        self.assertEqual(len(self.net.history), 50)

    def test_pipeline_integration(self):
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('net', self.net),
        ])
        pipe.fit(self.X, self.y)
        y_proba = pipe.predict_proba(self.X)
        self.assertEqual(y_proba.shape, (1000, 2))

    def test_grid_search(self):
        self.net.set_params(train_split=False, verbose=0)
        default_params = {
            'module__layer_sizes': [(12, 32, 20, 2)],
            'max_epochs': [50],
        }
        grid_search_params = {
            'module__dropout_percent': [.05, .2],
            **default_params,
        }
        gs = GridSearchCV(self.net, grid_search_params, n_jobs=1, refit=False, cv=2, scoring='accuracy', verbose=2)
        gs.fit(self.X, self.y)
        self.assertTrue(hasattr(gs, 'best_score_'))
        self.assertTrue(hasattr(gs, 'best_params_'))
        print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))


if __name__ == '__main__':
    unittest.main()