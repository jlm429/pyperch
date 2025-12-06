import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from pyperch import Trainer
from pyperch.config import TrainConfig, OptimizerConfig
from pyperch.core.metrics import MSE, R2


class SimpleMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    X, y = make_regression(
        n_samples=800,
        n_features=10,
        noise=10.0,
        random_state=42,
    )
    y = y.reshape(-1, 1)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=256)

    model = SimpleMLP(in_features=10)
    loss_fn = nn.MSELoss()

    opt_cfg = OptimizerConfig(
        name="sa",
        step_size=0.05,
        t=1.0,
        t_min=0.1,
        cooling=0.95,
    )

    cfg = TrainConfig(
        device="cpu",
        seed=42,
        max_epochs=50,
        optimizer="sa",
        optimizer_config=opt_cfg,
        optimizer_mode="per_epoch",
        metrics={
            "train": [MSE(), R2()],
            "valid": [MSE(), R2()],
        },
        callbacks=[],
    )

    trainer = Trainer(model, loss_fn, cfg)
    history = trainer.fit(train_loader, valid_loader)

    from pyperch.utils import plot_losses, plot_metrics
    plot_losses(history)
    plot_metrics(history, split="valid")

    print("Final train loss:", history["train_loss"][-1])
    print("Final valid loss:", history["valid_loss"][-1])


if __name__ == "__main__":
    main()
