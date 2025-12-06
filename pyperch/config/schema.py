from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.metrics import Metric
from ..core.callbacks import Callback


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters.

    These are deliberately generic; each optimizer will read only what it needs.
    """
    name: str = "sa"

    # Shared options
    step_size: float = 0.05
    inner_epochs: int = 1  # if an optimizer wants its own internal loop

    # SA-specific
    t: float = 1.0
    t_min: float = 0.1
    cooling: float = 0.95

    # GA-specific
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.5


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    # Device / reproducibility
    device: str = "cpu"
    seed: Optional[int] = None

    # Loop
    max_epochs: int = 100
    optimizer_mode: str = "per_epoch"  # or "per_batch"

    # Optimizer
    optimizer: str = "sa"
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Metrics: split -> list[Metric]
    metrics: Dict[str, List[Metric]] = field(default_factory=dict)

    # Callbacks
    callbacks: List[Callback] = field(default_factory=list)
    
    # layer options
    layer_modes: dict[str, str] | None = None   # {"layername": "freeze"|"grad"|"meta"}
