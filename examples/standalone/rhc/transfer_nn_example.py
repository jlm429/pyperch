"""
Train a model on digits, save it, reload it, freeze all but the last
3 layers, and fine-tune with PyPerch RHC.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyperch.optim import RHC

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.layers(x)


def evaluate(model, X, y, loss_fn):
    with torch.no_grad():
        output = model(X)
        loss = loss_fn(output, y).item()
        preds = output.argmax(dim=1)
        acc = (preds == y).float().mean().item()

    return loss, acc


# ------------------------------------------------------------
# Load digits dataset
# ------------------------------------------------------------
digits = load_digits()

X = digits.data.astype(np.float32)
y = digits.target.astype(np.int64)

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=seed,
    stratify=y,
)

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

loss_fn = nn.CrossEntropyLoss()


# ------------------------------------------------------------
# Step 1: Train model normally with Adam
#         Kept short so RHC still has room to improve.
# ------------------------------------------------------------
model = MyNetwork()

adam = torch.optim.Adam(model.parameters(), lr=0.001)

adam_train_accuracies = []
adam_test_accuracies = []

adam_epochs = 25

for epoch in range(adam_epochs):
    adam.zero_grad()

    loss = loss_fn(model(X_train), y_train)

    loss.backward()
    adam.step()

    train_loss, train_acc = evaluate(model, X_train, y_train, loss_fn)
    test_loss, test_acc = evaluate(model, X_test, y_test, loss_fn)

    adam_train_accuracies.append(train_acc)
    adam_test_accuracies.append(test_acc)

    print(
        f"adam epoch={epoch:03d} "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
    )


# ------------------------------------------------------------
# Step 2: Save pretrained model
# ------------------------------------------------------------
torch.save(model.state_dict(), "digits_model.pt")


# ------------------------------------------------------------
# Step 3: Load pretrained model
# ------------------------------------------------------------
model = MyNetwork()

model.load_state_dict(
    torch.load("digits_model.pt")
)

print("\nLoaded pretrained model.")


# ------------------------------------------------------------
# Step 4: Freeze all but the last 3 layers
#
# Last 3 layers are:
#   layer 4: Linear(64, 32)
#   layer 5: ReLU()
#   layer 6: Linear(32, 10)
#
# ReLU has no parameters, so effectively this trains
# the last two Linear layers.
# ------------------------------------------------------------
for layer in model.layers[:-3]:
    for param in layer.parameters():
        param.requires_grad = False

initial_first_layer_weight = model.layers[0].weight.detach().clone()


# ------------------------------------------------------------
# Step 5: Fine-tune remaining trainable layers with RHC
# ------------------------------------------------------------
optimizer = RHC(
    model.parameters(),
    step_size=0.02,
    random_state=seed,
)

rhc_train_accuracies = []
rhc_test_accuracies = []
rhc_step_losses = []


def closure():
    output = model(X_train)
    return loss_fn(output, y_train)


for epoch in range(1000):
    step_loss = optimizer.step(closure)

    train_loss, train_acc = evaluate(model, X_train, y_train, loss_fn)
    test_loss, test_acc = evaluate(model, X_test, y_test, loss_fn)

    rhc_step_losses.append(step_loss.item())
    rhc_train_accuracies.append(train_acc)
    rhc_test_accuracies.append(test_acc)

    if epoch % 25 == 0:
        print(
            f"rhc epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )


# ------------------------------------------------------------
# Step 6: Verify frozen layer did not change
# ------------------------------------------------------------
final_first_layer_weight = model.layers[0].weight.detach()

print(
    "Frozen first layer unchanged:",
    torch.allclose(initial_first_layer_weight, final_first_layer_weight)
)

print("Accepted steps:", optimizer.accepted_steps)
print("Rejected steps:", optimizer.rejected_steps)


# ------------------------------------------------------------
# Plot results
# ------------------------------------------------------------
plt.close("all")

plt.figure()
plt.plot(adam_train_accuracies, label="Train")
plt.plot(adam_test_accuracies, label="Validation")
plt.title("Initial Adam Training Learning Curve")
plt.xlabel("Adam Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(rhc_train_accuracies, label="Train")
plt.plot(rhc_test_accuracies, label="Validation")
plt.title("RHC Fine-Tuning Learning Curve")
plt.xlabel("RHC Iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(rhc_step_losses)
plt.title("RHC Optimizer Loss")
plt.xlabel("RHC Iteration")
plt.ylabel("Loss")
plt.show()