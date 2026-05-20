import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn

from pyperch.optim import SA

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

optimizer = SA(
    model.parameters(),
    step_size=0.05,
    temperature=2.0,
    min_temperature=0.001,
    cooling=0.995,
    random_state=seed,
)

train_losses = []
valid_losses = []
best_losses = []

train_r2s = []
valid_r2s = []

temperatures = []
acceptance_rates = []


def train_closure():
    output = model(X_train)
    return loss_fn(output, y_train)


def evaluate(X_data, y_data):
    with torch.no_grad():
        output = model(X_data)
        loss = loss_fn(output, y_data).item()

    r2 = r2_score(y_data.numpy(), output.numpy())

    return loss, r2


max_steps = 3000

for step in range(max_steps):
    optimizer.step(train_closure)

    train_loss, train_r2 = evaluate(X_train, y_train)
    valid_loss, valid_r2 = evaluate(X_valid, y_valid)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    best_losses.append(optimizer.best_loss)

    train_r2s.append(train_r2)
    valid_r2s.append(valid_r2)

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
            f"valid_r2={valid_r2:.4f} "
            f"temp={optimizer.temperature:.6f} "
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
print(f"Temperature   : {optimizer.temperature:.6f}")

print("\nBest Restored Model")
print("-" * 40)
print(f"Train loss : {best_train_loss:.6f}")
print(f"Valid loss : {best_valid_loss:.6f}")
print(f"Train R2   : {best_train_r2:.6f}")
print(f"Valid R2   : {best_valid_r2:.6f}")


# ------------------------------------------------------------
# Loss Curves
# ------------------------------------------------------------

plt.figure(figsize=(7, 4))

plt.plot(train_losses, label="current_train_loss")
plt.plot(valid_losses, label="current_valid_loss")
plt.plot(best_losses, label="best_train_loss")

plt.xlabel("step")
plt.ylabel("loss")
plt.title("SA Regression Loss")
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# R2 Curves
# ------------------------------------------------------------

plt.figure(figsize=(7, 4))

plt.plot(train_r2s, label="train_r2")
plt.plot(valid_r2s, label="valid_r2")

plt.xlabel("step")
plt.ylabel("R2")
plt.title("SA Regression R2")
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Temperature Curve
# ------------------------------------------------------------

plt.figure(figsize=(7, 4))

plt.plot(temperatures)

plt.yscale("log")

plt.xlabel("step")
plt.ylabel("temperature")
plt.title("SA Temperature")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Validation Loss vs Temperature
# ------------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(7, 4))

ax1.plot(
    valid_losses,
    label="validation_loss",
)

ax1.set_xlabel("step")
ax1.set_ylabel("validation loss")

ax2 = ax1.twinx()

ax2.plot(
    temperatures,
    linestyle="--",
    label="temperature",
)

ax2.set_yscale("log")
ax2.set_ylabel("temperature")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2)

plt.title("SA Temperature vs Validation Loss")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Acceptance Rate
# ------------------------------------------------------------

plt.figure(figsize=(7, 4))

plt.plot(acceptance_rates)

plt.xlabel("step")
plt.ylabel("acceptance rate")
plt.title("SA Acceptance Rate")
plt.tight_layout()
plt.show()
