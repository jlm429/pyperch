import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn

from pyperch.optim import SA

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

X, y = make_classification(
    n_samples=1000,
    n_features=12,
    n_informative=10,
    n_classes=2,
    random_state=seed,
)

X = X.astype(np.float32)
y = y.astype(np.int64)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=seed,
)

X_train = torch.tensor(X_train)
X_valid = torch.tensor(X_valid)
y_train = torch.tensor(y_train)
y_valid = torch.tensor(y_valid)


model = nn.Sequential(
    nn.Linear(12, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

loss_fn = nn.CrossEntropyLoss()

optimizer = SA(
    model.parameters(),
    step_size=0.025,
    temperature=1.0,
    min_temperature=0.001,
    cooling=0.995,
    random_state=seed,
)

train_losses = []
valid_losses = []
best_losses = []
train_f1s = []
valid_f1s = []
temperatures = []
acceptance_rates = []


def train_closure():
    output = model(X_train)
    return loss_fn(output, y_train)


def evaluate(X_data, y_data):
    with torch.no_grad():
        output = model(X_data)
        loss = loss_fn(output, y_data).item()
        preds = output.argmax(dim=1)

    f1 = f1_score(y_data.numpy(), preds.numpy(), average="binary")
    return loss, f1


max_steps = 3000

for step in range(max_steps):
    optimizer.step(train_closure)

    train_loss, train_f1 = evaluate(X_train, y_train)
    valid_loss, valid_f1 = evaluate(X_valid, y_valid)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    best_losses.append(optimizer.best_loss)
    train_f1s.append(train_f1)
    valid_f1s.append(valid_f1)
    temperatures.append(optimizer.temperature)

    if optimizer.proposed_steps > 0:
        acceptance_rate = optimizer.accepted_steps / optimizer.proposed_steps
    else:
        acceptance_rate = 0.0

    acceptance_rates.append(acceptance_rate)

    if step % 250 == 0:
        print(
            f"step={step:04d} "
            f"train_loss={train_loss:.4f} "
            f"best_loss={optimizer.best_loss:.4f} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_f1={valid_f1:.4f} "
            f"temp={optimizer.temperature:.6f} "
            f"accepted={optimizer.accepted_steps} "
            f"rejected={optimizer.rejected_steps}"
        )

optimizer.restore_best()

best_train_loss, best_train_f1 = evaluate(X_train, y_train)
best_valid_loss, best_valid_f1 = evaluate(X_valid, y_valid)

print("\nOptimizer Summary")
print("-" * 40)
print(f"Function evals : {optimizer.function_evals}")
print(f"Proposed steps: {optimizer.proposed_steps}")
print(f"Accepted steps: {optimizer.accepted_steps}")
print(f"Rejected steps: {optimizer.rejected_steps}")
print(f"Best loss     : {optimizer.best_loss:.6f}")
print(f"Temperature   : {optimizer.temperature:.6f}")
print(f"Best train loss after restore: {best_train_loss:.6f}")
print(f"Best valid loss after restore: {best_valid_loss:.6f}")
print(f"Best train F1 after restore  : {best_train_f1:.6f}")
print(f"Best valid F1 after restore  : {best_valid_f1:.6f}")

plt.figure()
plt.plot(train_losses, label="current_train_loss")
plt.plot(valid_losses, label="current_valid_loss")
plt.plot(best_losses, label="best_train_loss")
plt.legend()
plt.title("SA Classification Loss")
plt.show()

plt.figure()
plt.plot(train_f1s, label="current_train_f1")
plt.plot(valid_f1s, label="current_valid_f1")
plt.legend()
plt.title("SA Classification F1")
plt.show()

plt.figure()
plt.plot(temperatures)
plt.yscale("log")
plt.title("SA Temperature")
plt.show()

plt.figure()
plt.plot(acceptance_rates)
plt.title("SA Acceptance Rate")
plt.show()
