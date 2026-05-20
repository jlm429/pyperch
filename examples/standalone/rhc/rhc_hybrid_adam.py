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
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)

loss_fn = nn.CrossEntropyLoss()

for param in model[0].parameters():
    param.requires_grad = False

initial_frozen_weight = model[0].weight.detach().clone()

adam_params = list(model[2].parameters())
rhc_params = list(model[4].parameters())

adam = torch.optim.Adam(adam_params, lr=1e-3)

rhc = RHC(
    rhc_params,
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


for epoch in range(20):
    adam.zero_grad()
    loss = closure()
    loss.backward()
    adam.step()

    rhc.step(closure)

    eval_loss, acc = evaluate()
    losses.append(eval_loss)
    accuracies.append(acc)

    print(f"epoch={epoch:03d} loss={eval_loss:.4f} accuracy={acc:.4f}")

final_frozen_weight = model[0].weight.detach()

print(
    "Frozen layer unchanged:",
    torch.allclose(initial_frozen_weight, final_frozen_weight),
)
print("RHC function evals:", rhc.function_evals)
print("RHC proposed steps:", rhc.proposed_steps)
print("RHC accepted steps:", rhc.accepted_steps)
print("RHC rejected steps:", rhc.rejected_steps)
print("RHC best loss:", rhc.best_loss)

plt.figure()
plt.plot(losses)
plt.title("Hybrid Adam + RHC Loss")
plt.show()

plt.figure()
plt.plot(accuracies)
plt.title("Hybrid Adam + RHC Accuracy")
plt.show()
