import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn

from pyperch.optim import RHC

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

optimizer = RHC(
    model.parameters(),
    step_size=0.05,
    restarts=3,
    restart_interval=100,
    random_state=seed,
)

train_losses = []
valid_losses = []
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


for epoch in range(250):
    train_loss = optimizer.step(train_closure)

    train_eval_loss, train_acc = evaluate(X_train, y_train)
    valid_loss, valid_acc = evaluate(X_valid, y_valid)

    train_losses.append(train_eval_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)

    if epoch % 25 == 0:
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_eval_loss:.4f} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_acc={valid_acc:.4f}"
        )

optimizer.restore_best()

print("Function evals:", optimizer.function_evals)
print("Proposed steps:", optimizer.proposed_steps)
print("Accepted steps:", optimizer.accepted_steps)
print("Rejected steps:", optimizer.rejected_steps)
print("Best loss:", optimizer.best_loss)

plt.figure()
plt.plot(train_losses, label="train_loss")
plt.plot(valid_losses, label="valid_loss")
plt.legend()
plt.title("RHC Classification Loss")
plt.show()

plt.figure()
plt.plot(train_accs, label="train_accuracy")
plt.plot(valid_accs, label="valid_accuracy")
plt.legend()
plt.title("RHC Classification Accuracy")
plt.show()
