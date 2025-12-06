from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch

from ..config.schema import TrainConfig


@dataclass
class Engine:
    model: torch.nn.Module
    config: TrainConfig

    epoch: int = 0
    step: int = 0

    train_loss: float | None = None
    valid_loss: float | None = None

    metric_values: Dict[str, Dict[str, float]] = field(default_factory=dict)

    stop_training: bool = False

    def set_metrics(self, split: str, values: Dict[str, float]) -> None:
        self.metric_values[split] = values
