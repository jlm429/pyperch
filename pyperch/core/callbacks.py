from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


class Callback:
    def on_train_begin(self, engine: "Engine") -> None: ...
    def on_epoch_begin(self, engine: "Engine") -> None: ...
    def on_step_end(self, engine: "Engine") -> None: ...
    def on_epoch_end(self, engine: "Engine") -> None: ...
    def on_train_end(self, engine: "Engine") -> None: ...


class CaptureInitialWeights(Callback):
    """
    Captures the initial weights of specified parameters at the start of training.
    Useful for verifying freeze() and meta_opt() behavior.
    """

    def __init__(self, *param_names: str):
        self.param_names = param_names
        self.initial = {}

    def on_train_begin(self, engine: "Engine") -> None:
        model = engine.model
        for name, p in model.named_parameters():
            if name in self.param_names:
                self.initial[name] = p.detach().clone()


@dataclass
class HistoryCallback(Callback):
    history: Dict[str, Any] = field(
        default_factory=lambda: {
            "epoch": [],
            "train_loss": [],
            "valid_loss": [],
            "train_metrics": {},
            "valid_metrics": {},
        }
    )

    def on_epoch_end(self, engine: "Engine") -> None:
        h = self.history
        h["epoch"].append(engine.epoch)
        h["train_loss"].append(engine.train_loss)
        h["valid_loss"].append(engine.valid_loss)
        for split in ("train", "valid"):
            metrics = engine.metric_values.get(split, {})
            key = f"{split}_metrics"
            if key not in h:
                h[key] = {}
            for name, value in metrics.items():
                h[key].setdefault(name, []).append(value)


@dataclass
class EarlyStopping(Callback):
    monitor: str = "valid_loss"
    patience: int = 10
    min_delta: float = 0.0

    best: float | None = None
    wait: int = 0
    stopped_epoch: int | None = None

    def on_epoch_end(self, engine: "Engine") -> None:
        current = getattr(engine, self.monitor, None)
        if current is None:
            return
        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = engine.epoch
                engine.stop_training = True


class CallbackList(Callback):
    def __init__(self, callbacks: List[Callback] | None = None):
        self.callbacks = callbacks or []

    def append(self, cb: Callback) -> None:
        self.callbacks.append(cb)

    def on_train_begin(self, engine: "Engine") -> None:
        for cb in self.callbacks:
            cb.on_train_begin(engine)

    def on_epoch_begin(self, engine: "Engine") -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(engine)

    def on_step_end(self, engine: "Engine") -> None:
        for cb in self.callbacks:
            cb.on_step_end(engine)

    def on_epoch_end(self, engine: "Engine") -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(engine)

    def on_train_end(self, engine: "Engine") -> None:
        for cb in self.callbacks:
            cb.on_train_end(engine)
