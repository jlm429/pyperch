import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn

from pyperch.optim import SA
from pyperch.search import OptunaSearch

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Model factory
# ------------------------------------------------------------
# Each Optuna trial needs a fresh model.
# ------------------------------------------------------------
def make_model():
    return nn.Sequential(
        nn.Linear(12, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )


loss_fn = nn.CrossEntropyLoss()

# ------------------------------------------------------------
# Define the Optuna search space
# ------------------------------------------------------------
# Each parameter is sampled automatically for every trial.
#
# Format:
# ("float", low, high, log_scale)
# ("int", low, high, step)
# ("categorical", [choices])
# ------------------------------------------------------------
search = OptunaSearch(
    param_space={
        "step_size": ("float", 0.01, 0.5, True),
        "temperature": ("float", 0.1, 2.0, True),
        "cooling": ("float", 0.90, 0.999),
        "min_temperature": ("float", .001, .01, True),
    },
    direction="minimize",
)



# ------------------------------------------------------------
# Objective function
# ------------------------------------------------------------
# Optuna calls this once per trial.
#
# `params` contains the sampled hyperparameters.
# `trial` is the native Optuna trial object.
#
# The objective should:
# 1. Create a fresh model
# 2. Create the optimizer using sampled params
# 3. Train the model
# 4. Return a scalar validation metric
# ------------------------------------------------------------
def objective(params, trial):
    torch.manual_seed(seed)

    model = make_model()

    optimizer = SA(
        model.parameters(),
        step_size=params["step_size"],
        temperature=params["temperature"],
        cooling=params["cooling"],
        min_temperature=params["min_temperature"],
    )

    for _ in range(3000):

        def closure():
            output = model(X_train)
            return loss_fn(output, y_train)

        optimizer.step(closure)

    with torch.no_grad():
        valid_loss = loss_fn(model(X_valid), y_valid).item()

    return valid_loss


# ------------------------------------------------------------
# Run search
# ------------------------------------------------------------
study = search.search(
    objective,
    n_trials=20,
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
print("Best parameters:")
print(study.best_params)

print("\nBest validation loss:")
print(study.best_value)

# ------------------------------------------------------------
# Retrain using best parameters and track learning curve
# ------------------------------------------------------------
best_params = study.best_params

torch.manual_seed(seed)
best_model = make_model()

best_optimizer = SA(
    best_model.parameters(),
    step_size=best_params["step_size"],
    temperature=best_params["temperature"],
    cooling=best_params["cooling"],
    min_temperature=best_params["min_temperature"],
)

train_losses = []
valid_losses = []

for _ in range(3000):

    def closure():
        output = best_model(X_train)
        return loss_fn(output, y_train)

    loss = best_optimizer.step(closure)
    train_losses.append(loss.item() if torch.is_tensor(loss) else loss)

    with torch.no_grad():
        valid_loss = loss_fn(best_model(X_valid), y_valid).item()
        valid_losses.append(valid_loss)


# ------------------------------------------------------------
# Plot learning curve
# ------------------------------------------------------------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("SA Learning Curve Using Best Optuna Parameters")
plt.legend()
plt.show()