import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from pyperch.optim import RHC

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

X = np.random.randn(500, 10).astype(np.float32)
y = (X[:, :3].sum(axis=1) > 0).astype(np.int64)

X = torch.tensor(X)
y = torch.tensor(y)


model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)

for param in model[0].parameters():
    param.requires_grad = False

initial_weight = model[0].weight.detach().clone()

loss_fn = nn.CrossEntropyLoss()

optimizer = RHC(
    model.parameters(),
    step_size=0.5,
    random_state=seed,
)

losses = []
accuracies = []


def closure():
    output = model(X)
    return loss_fn(output, y)


def evaluate():
    with torch.no_grad():
        output = model(X)
        loss = loss_fn(output, y).item()
        preds = output.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return loss, acc


for epoch in range(200):
    optimizer.step(closure)

    loss, acc = evaluate()
    losses.append(loss)
    accuracies.append(acc)

    if epoch % 25 == 0:
        print(f"epoch={epoch:03d} loss={loss:.4f} accuracy={acc:.4f}")

final_weight = model[0].weight.detach()

print("Frozen layer unchanged:", torch.allclose(initial_weight, final_weight))
print("Accepted steps:", optimizer.accepted_steps)
print("Rejected steps:", optimizer.rejected_steps)

plt.figure()
plt.plot(losses)
plt.title("RHC Loss With Frozen First Layer")
plt.show()

plt.figure()
plt.plot(accuracies)
plt.title("RHC Accuracy With Frozen First Layer")
plt.show()
