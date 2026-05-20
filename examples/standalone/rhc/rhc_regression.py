import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn

from pyperch.optim import RHC

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

X, y = make_regression(
    n_samples=1000,
    n_features=12,
    n_informative=10,
    noise=0.2,
    random_state=seed,
)

X = X.astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=seed,
)

X_train = torch.tensor(X_train)
X_valid = torch.tensor(X_valid)
y_train = torch.tensor(y_train)
y_valid = torch.tensor(y_valid)


model = nn.Sequential(
    nn.Linear(12, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 1),
)

loss_fn = nn.MSELoss()

optimizer = RHC(
    model.parameters(),
    step_size=0.5,
    restarts=3,
    restart_interval=150,
    random_state=seed,
)

train_losses = []
valid_losses = []
train_r2s = []
valid_r2s = []


def train_closure():
    output = model(X_train)
    return loss_fn(output, y_train)


def evaluate(X_data, y_data):
    with torch.no_grad():
        output = model(X_data)
        loss = loss_fn(output, y_data).item()
        r2 = r2_score(y_data.numpy(), output.numpy())
    return loss, r2


for epoch in range(500):
    optimizer.step(train_closure)

    train_loss, train_r2 = evaluate(X_train, y_train)
    valid_loss, valid_r2 = evaluate(X_valid, y_valid)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_r2s.append(train_r2)
    valid_r2s.append(valid_r2)

    if epoch % 50 == 0:
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_r2={valid_r2:.4f}"
        )

optimizer.restore_best()

print("Final train loss:", train_losses[-1])
print("Final valid loss:", valid_losses[-1])
print("Final train R2:", train_r2s[-1])
print("Final valid R2:", valid_r2s[-1])

print("Function evals:", optimizer.function_evals)
print("Proposed steps:", optimizer.proposed_steps)
print("Accepted steps:", optimizer.accepted_steps)
print("Rejected steps:", optimizer.rejected_steps)
print("Best loss:", optimizer.best_loss)

plt.figure()
plt.plot(train_losses, label="train_loss")
plt.plot(valid_losses, label="valid_loss")
plt.legend()
plt.title("RHC Regression Loss")
plt.show()

plt.figure()
plt.plot(train_r2s, label="train_r2")
plt.plot(valid_r2s, label="valid_r2")
plt.legend()
plt.title("RHC Regression R2")
plt.show()
