import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn

from pyperch.optim import GA

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

optimizer = GA(
    model.parameters(),
    population_size=250,
    mutation_rate=0.05,
    step_size=0.05,
    random_state=seed,
)

train_losses = []
valid_losses = []
best_losses = []
train_accs = []
valid_accs = []


def train_closure():
    output = model(X_train)
    return loss_fn(output, y_train)


def evaluate(X_data, y_data):
    with torch.no_grad():
        output = model(X_data)
        loss = loss_fn(output, y_data).item()
        preds = output.argmax(dim=1)
        acc = (preds == y_data).float().mean().item()

    return loss, acc


max_steps = 40

for step in range(max_steps):
    optimizer.step(train_closure)

    train_loss, train_acc = evaluate(X_train, y_train)
    valid_loss, valid_acc = evaluate(X_valid, y_valid)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    best_losses.append(optimizer.best_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)

    print(
        f"step={step:03d} "
        f"train_loss={train_loss:.4f} "
        f"best_loss={optimizer.best_loss:.4f} "
        f"valid_loss={valid_loss:.4f} "
        f"valid_acc={valid_acc:.4f} "
        f"accepted={optimizer.accepted_steps} "
        f"rejected={optimizer.rejected_steps}"
    )

optimizer.restore_best()

best_train_loss, best_train_acc = evaluate(X_train, y_train)
best_valid_loss, best_valid_acc = evaluate(X_valid, y_valid)

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
print(f"Train acc  : {best_train_acc:.6f}")
print(f"Valid acc  : {best_valid_acc:.6f}")

plt.figure(figsize=(7, 4))
plt.plot(train_losses, label="current_train_loss")
plt.plot(valid_losses, label="current_valid_loss")
plt.plot(best_losses, label="best_train_loss")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("GA Classification Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(train_accs, label="train_accuracy")
plt.plot(valid_accs, label="valid_accuracy")
plt.xlabel("step")
plt.ylabel("accuracy")
plt.title("GA Classification Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
