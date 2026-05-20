import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn

from pyperch.optim import GA

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

optimizer = GA(
    model.parameters(),
    population_size=60,
    mutation_rate=0.15,
    step_size=0.1,
    random_state=seed,
)

train_losses = []
valid_losses = []
best_losses = []
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


max_steps = 20

for step in range(max_steps):
    optimizer.step(train_closure)

    train_loss, train_r2 = evaluate(X_train, y_train)
    valid_loss, valid_r2 = evaluate(X_valid, y_valid)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    best_losses.append(optimizer.best_loss)
    train_r2s.append(train_r2)
    valid_r2s.append(valid_r2)

    print(
        f"step={step:03d} "
        f"train_loss={train_loss:.4f} "
        f"best_loss={optimizer.best_loss:.4f} "
        f"valid_loss={valid_loss:.4f} "
        f"valid_r2={valid_r2:.4f} "
        f"accepted={optimizer.accepted_steps} "
        f"rejected={optimizer.rejected_steps}"
    )

optimizer.restore_best()

best_train_loss, best_train_r2 = evaluate(X_train, y_train)
best_valid_loss, best_valid_r2 = evaluate(X_valid, y_valid)

print("\nOptimizer Summary")
print("-" * 40)
print(f"Function evals : {optimizer.function_evals}")
print(f"Proposed steps: {optimizer.proposed_steps}")
print(f"Accepted steps: {optimizer.accepted_steps}")
print(f"Rejected steps: {optimizer.rejected_steps}")
print(f"Best loss     : {optimizer.best_loss:.6f}")

print("\nBest Restored Model")
print("-" * 40)
print(f"Train loss : {best_train_loss:.6f}")
print(f"Valid loss : {best_valid_loss:.6f}")
print(f"Train R2   : {best_train_r2:.6f}")
print(f"Valid R2   : {best_valid_r2:.6f}")

plt.figure(figsize=(7, 4))
plt.plot(train_losses, label="current_train_loss")
plt.plot(valid_losses, label="current_valid_loss")
plt.plot(best_losses, label="best_train_loss")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("GA Regression Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(train_r2s, label="train_r2")
plt.plot(valid_r2s, label="valid_r2")
plt.xlabel("step")
plt.ylabel("R2")
plt.title("GA Regression R2")
plt.legend()
plt.tight_layout()
plt.show()
