from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch


class Metric(ABC):
    """Simple streaming metric API: reset -> update -> compute."""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        ...

    @abstractmethod
    def compute(self) -> float:
        ...


class MSE(Metric):
    def __init__(self):
        super().__init__(name="mse")

    def reset(self) -> None:
        self._sum_sq = 0.0
        self._n = 0

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        diff = preds.detach() - target.detach()
        self._sum_sq += float((diff ** 2).mean().item())
        self._n += 1

    def compute(self) -> float:
        return self._sum_sq / max(self._n, 1)


class R2(Metric):
    def __init__(self):
        super().__init__(name="r2")

    def reset(self) -> None:
        self._ss_res = 0.0
        self._ss_tot = 0.0
        self._mean_y = 0.0
        self._n = 0

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        y = target.detach().view(-1)
        y_hat = preds.detach().view(-1)
        if self._n == 0:
            self._mean_y = float(y.mean().item())
        self._ss_res += float(((y - y_hat) ** 2).sum().item())
        self._ss_tot += float(((y - self._mean_y) ** 2).sum().item())
        self._n += int(y.numel())

    def compute(self) -> float:
        if self._ss_tot == 0.0:
            return 0.0
        return 1.0 - (self._ss_res / self._ss_tot)


class Accuracy(Metric):
    def __init__(self):
        super().__init__(name="accuracy")

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.ndim > 1 and preds.size(-1) > 1:
            y_hat = preds.argmax(dim=-1)
        else:
            y_hat = (preds.view(-1) >= 0.5).long()
        y = target.view_as(y_hat).long()
        self._correct += int((y_hat == y).sum().item())
        self._total += int(y.numel())

    def compute(self) -> float:
        if self._total == 0:
            return 0.0
        return self._correct / self._total


BUILTIN_METRICS: Dict[str, type[Metric]] = {
    "mse": MSE,
    "r2": R2,
    "accuracy": Accuracy,
}
