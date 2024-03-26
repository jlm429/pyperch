# modified from skorch notebook
# https://nbviewer.org/github/skorch-dev/skorch/blob/master/notebooks/Basic_Usage.ipynb

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_classification
from skorch import NeuralNetClassifier
from skorch import NeuralNet
from pyperch.utils.decorators import add_to

#set seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# This is a toy dataset for binary classification, 1000 data points with 20 features each
X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X, y = X.astype(np.float32), y.astype(np.int64)

print(X.shape, y.shape, y.mean())

class ClassifierModule(nn.Module):
    def __init__(
            self,
            num_units=10,
            nonlin=F.relu,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin
        self.dropout = dropout

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X

    def do_something(self):
        #current_weights = copy.deepcopy(self.model.weight)
        print("Doing something")

net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=20,
    lr=0.1,
#     device='cuda',  # uncomment this to train with CUDA
)

@add_to(NeuralNet)
def test_func(self):
    print("using add to")

#override fit method
@add_to(NeuralNet)
def fit(self, X, y=None, **fit_params):
    if not self.warm_start or not self.initialized_:
        self.initialize()
    print("calling partial_fit")
    self.partial_fit(X, y, **fit_params)
    return self

# Training the network
net.fit(X, y)
model = net.module_
print(model.do_something())
for param in model.parameters():
    print("model weights:", param)

# Making prediction for first 5 data points of X
y_pred = net.predict(X[:5])
print(y_pred)

# Checking probarbility of each class for first 5 data points of X
y_proba = net.predict_proba(X[:5])
print(y_proba)