from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any, List, Sequence, Tuple

import fnmatch
import numpy as np
import torch
from torch import nn

from .config import TrainConfig, OptimizerConfig, TorchConfig, ModelConfig
from .core.metrics import Metric, BUILTIN_METRICS
from .core.callbacks import Callback
from .core.trainer import Trainer
from .models import SimpleMLP
from .utils import make_loaders


class Perch:
    """
    Builder for configuring and running Pyperch experiments.
    """

    def __init__(self) -> None:
        self._model_cls: type[nn.Module] | None = None
        self._model_kwargs: dict[str, Any] = {}
        self._loss_fn: nn.Module | None = None

        self._optimizer_name: str | None = None
        self._optimizer_kwargs: dict[str, Any] = {}

        self._torch_optimizer_name: str | None = None
        self._torch_optimizer_kwargs: dict[str, Any] = {}

        self._metric_specs: list[Metric | type[Metric] | str] = []

        self._X: Any = None
        self._y: Any = None
        self._data_kwargs: dict[str, Any] = {}

        self._train_loader = None
        self._valid_loader = None

        self._callbacks: list[Callback] = []

        self._layer_modes: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------
    def model(
        self,
        model_cls: type[nn.Module],
        /,
        *,
        loss_fn: nn.Module | None = None,
        **model_kwargs: Any,
    ) -> "Perch":
        self._model_cls = model_cls
        self._model_kwargs = model_kwargs
        if loss_fn is not None:
            self._loss_fn = loss_fn
        return self

    def simple_classifier(
        self,
        X,
        y,
        *,
        hidden: Sequence[int] | None = None,
        activation: str = "relu",
        batch_size: int = 64,
        valid_split: float = 0.2,
        stratify: bool = True,
        normalize: str | None = None,
    ) -> "Perch":
        if hidden is None:
            hidden = [32]

        X_arr = X if isinstance(X, np.ndarray) else np.asarray(X)
        input_dim = X_arr.shape[1]

        if isinstance(y, torch.Tensor):
            n_classes = int(torch.unique(y).numel())
        else:
            n_classes = int(np.unique(y).size)

        self._model_cls = SimpleMLP
        self._model_kwargs = {
            "input_dim": input_dim,
            "hidden": list(hidden),
            "output_dim": n_classes,
            "activation": activation,
        }
        self._loss_fn = nn.CrossEntropyLoss()

        if not self._metric_specs:
            self._metric_specs = ["accuracy"]

        if self._optimizer_name is None:
            self._optimizer_name = "rhc"

        self._X = X
        self._y = y
        self._data_kwargs = {
            "batch_size": batch_size,
            "valid_split": valid_split,
            "stratify": stratify,
            "normalize": normalize,
        }
        return self

    # ------------------------------------------------------------------
    # Meta-optimizer (SA / GA / RHC)
    # ------------------------------------------------------------------
    def optimizer(self, name: str, /, **kwargs: Any) -> "Perch":
        self._optimizer_name = name
        self._optimizer_kwargs.update(kwargs)
        return self

    # ------------------------------------------------------------------
    # PyTorch optimizer
    # ------------------------------------------------------------------
    def torch_optimizer(self, name: str = "adam", /, **kwargs: Any) -> "Perch":
        self._torch_optimizer_name = name
        self._torch_optimizer_kwargs.update(kwargs)
        return self

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def metrics(self, *metrics: Metric | type[Metric] | str) -> "Perch":
        self._metric_specs = list(metrics)
        return self

    # ------------------------------------------------------------------
    # Data configuration
    # ------------------------------------------------------------------
    def data(self, X, y, **loader_kwargs: Any) -> "Perch":
        self._X = X
        self._y = y
        self._data_kwargs = loader_kwargs
        return self

    def data_loaders(self, train_loader, valid_loader=None) -> "Perch":
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        return self

    # ------------------------------------------------------------------
    # Layer modes (freeze / grad / meta) using wildcard patterns
    # ------------------------------------------------------------------
    def freeze(self, *patterns: str) -> "Perch":
        self._layer_modes.update({p: "freeze" for p in patterns})
        return self

    def grad_opt(self, *patterns: str) -> "Perch":
        self._layer_modes.update({p: "grad" for p in patterns})
        return self

    def meta_opt(self, *patterns: str) -> "Perch":
        self._layer_modes.update({p: "meta" for p in patterns})
        return self

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def callbacks(self, *callbacks: Callback) -> "Perch":
        self._callbacks.extend(callbacks)
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_metrics(self) -> dict[str, List[Metric]]:
        if not self._metric_specs:
            return {}

        metric_classes: list[type[Metric]] = []
        for spec in self._metric_specs:
            if isinstance(spec, str):
                cls = BUILTIN_METRICS.get(spec)
                if cls is None:
                    raise ValueError(f"Unknown metric '{spec}'")
                metric_classes.append(cls)
            elif isinstance(spec, Metric):
                metric_classes.append(spec.__class__)
            elif isinstance(spec, type) and issubclass(spec, Metric):
                metric_classes.append(spec)
            else:
                raise TypeError(f"Unsupported metric spec: {spec}")

        return {
            "train": [cls() for cls in metric_classes],
            "valid": [cls() for cls in metric_classes],
        }

    def _build_optimizer_config(self) -> OptimizerConfig:
        name = self._optimizer_name or "sa"
        valid = {f.name for f in dataclass_fields(OptimizerConfig)}
        kwargs = {k: v for k, v in self._optimizer_kwargs.items() if k in valid}
        return OptimizerConfig(name=name, **kwargs)

    def _build_torch_config(self) -> TorchConfig | None:
        if self._torch_optimizer_name is None and not self._torch_optimizer_kwargs:
            return None
        valid = {f.name for f in dataclass_fields(TorchConfig)}
        kws = dict(self._torch_optimizer_kwargs)
        kws["optimizer"] = self._torch_optimizer_name or "adam"
        clean = {k: v for k, v in kws.items() if k in valid}
        return TorchConfig(**clean)

    def _ensure_loaders(self, seed: int | None):
        if self._train_loader is not None:
            return self._train_loader, self._valid_loader
        if self._X is None or self._y is None:
            raise ValueError("No data provided.")
        kws = dict(self._data_kwargs)
        if "seed" not in kws and seed is not None:
            kws["seed"] = seed
        return make_loaders(self._X, self._y, **kws)

    # ------------------------------------------------------------------
    # Build model for inspection (e.g., freezing tests)
    # ------------------------------------------------------------------
    def build_model(self) -> nn.Module:
        if self._model_cls is None:
            raise ValueError("Model not set.")

        model = self._model_cls(**self._model_kwargs)

        # Apply layer modes exactly like in train()
        if self._layer_modes:
            for pattern, mode in self._layer_modes.items():
                for name, p in model.named_parameters():
                    if fnmatch.fnmatch(name, pattern):
                        if mode == "freeze":
                            p.requires_grad = False
                        elif mode in ("grad", "meta"):
                            p.requires_grad = True

        return model

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train(
        self,
        *,
        max_epochs: int | None = None,
        optimizer_mode: str | None = None,
        seed: int | None = None,
        device: str = "cpu",
    ) -> tuple[Trainer, dict]:
        if self._model_cls is None:
            raise ValueError("Model not set.")

        model = self._model_cls(**self._model_kwargs)

        # Apply freeze/grad/meta after constructing the model
        if self._layer_modes:
            for pattern, mode in self._layer_modes.items():
                for name, p in model.named_parameters():
                    if fnmatch.fnmatch(name, pattern):
                        if mode == "freeze":
                            p.requires_grad = False
                        elif mode == "grad":
                            p.requires_grad = True
                        elif mode == "meta":
                            p.requires_grad = True

        if self._loss_fn is None:
            if hasattr(model, "net") and isinstance(model.net[-1], nn.Linear):
                out_dim = model.net[-1].out_features
                self._loss_fn = nn.MSELoss() if out_dim == 1 else nn.CrossEntropyLoss()
            else:
                self._loss_fn = nn.MSELoss()

        loss_fn = self._loss_fn

        train_loader, valid_loader = self._ensure_loaders(seed)

        metrics_dict = self._build_metrics()
        opt_cfg = self._build_optimizer_config()
        torch_cfg = self._build_torch_config()

        if max_epochs is None:
            max_epochs = 100
        if optimizer_mode is None:
            optimizer_mode = "per_epoch"

        train_cfg = TrainConfig(
            device=device,
            seed=seed,
            max_epochs=max_epochs,
            optimizer_mode=optimizer_mode,
            optimizer=self._optimizer_name or "sa",
            optimizer_config=opt_cfg,
            torch_config=torch_cfg,
            model_config=None,
            metrics=metrics_dict,
            callbacks=self._callbacks,
            layer_modes=self._layer_modes or None,
        )

        trainer = Trainer(model=model, loss_fn=loss_fn, config=train_cfg)
        history = trainer.fit(train_loader=train_loader, valid_loader=valid_loader)
        return trainer, history
