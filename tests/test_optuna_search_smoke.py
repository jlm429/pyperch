import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyperch import Trainer
from pyperch.config import TrainConfig, OptimizerConfig, TorchConfig
from pyperch.core.metrics import Accuracy
from pyperch.models import SimpleMLP

from pyperch.search.strategy import OptunaStrategy
from pyperch.search.builder import TrainConfigBuilder
from pyperch.search.adapter import TrainerAdapter
from pyperch.search.factory import SearchFactory


def test_optuna_smoke_test_one_trial():
    np.random.seed(0)
    torch.manual_seed(0)

    X = torch.randn(60, 10)
    y = (X.sum(dim=1) > 0).long()

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=16)

    # Base config
    opt_cfg = OptimizerConfig(name="sa", t=1.0, t_min=0.1, step_size=0.1, cooling=0.95)

    base_cfg = TrainConfig(
        device="cpu",
        seed=0,
        max_epochs=1,
        optimizer="sa",
        optimizer_config=opt_cfg,
        metrics={"train": [Accuracy()], "valid": [Accuracy()]},
        torch_config=TorchConfig(),
    )

    builder = TrainConfigBuilder(base_cfg)

    # Suggestion with minimal space
    def suggest_params(trial):
        return {
            "optimizer_config.t": trial.suggest_float("t", 0.5, 1.5),
            "max_epochs": 1,
        }

    strategy = OptunaStrategy(suggest_params)

    def train_fn(cfg):
        model = SimpleMLP(input_dim=10, hidden=[8], output_dim=2)
        trainer = Trainer(model, nn.CrossEntropyLoss(), cfg)
        history = trainer.fit(loader, loader)
        return history["train_metrics"]["accuracy"][-1]

    adapter = TrainerAdapter(builder, strategy, train_fn)

    # In-memory SQLite is allowed
    search = SearchFactory.optuna_sqlite(
        adapter=adapter,
        study_name="test_search",
        storage="sqlite:///test_search.db",
    )

    result = search.run(n_trials=1, n_jobs=1)

    # Should return a float score
    assert isinstance(search.best_value, float)
