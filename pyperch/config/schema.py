from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.metrics import Metric
from ..core.callbacks import Callback


# ----------------------------------------------------------
# Meta-optimizer config (RHC, SA, GA)
# ----------------------------------------------------------
@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters for SA / GA / RHC."""

    name: str = "sa"

    # Shared options
    step_size: float = 0.05
    inner_epochs: int = 1

    # SA-specific
    t: float = 1.0
    t_min: float = 0.1
    cooling: float = 0.95

    # GA-specific
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.5


# ----------------------------------------------------------
# PyTorch optimizer configuration
# ----------------------------------------------------------
@dataclass
class TorchConfig:
    """Optional PyTorch optimizer configuration.
    defaults to Adam(lr=1e-3).
    """

    optimizer: str = "adam"  # adam | sgd | rmsprop
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Adam params
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # SGD params
    momentum: float = 0.0
    nesterov: bool = False

    # RMSprop params
    alpha: float = 0.99
    centered: bool = False


# ----------------------------------------------------------
# Model configuration (activation, etc.)
# ----------------------------------------------------------
@dataclass
class ModelConfig:
    """Model options such as activation function.
    Defaults preserve existing behavior.
    """

    activation: str = "relu"  # relu | leaky_relu | tanh | sigmoid


# ----------------------------------------------------------
# TrainConfig
# ----------------------------------------------------------
@dataclass
class TrainConfig:
    """Top-level training configuration."""

    # Device / reproducibility
    device: str = "cpu"
    seed: Optional[int] = None

    # Loop
    max_epochs: int = 100
    optimizer_mode: str = "per_epoch"  # or "per_batch"

    # Meta-optimizer (SA/GA/RHC)
    optimizer: str = "sa"
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)

    # optional PyTorch optimizer settings
    torch_config: Optional[TorchConfig] = None

    # optional model settings (e.g., activation)
    model_config: Optional[ModelConfig] = None

    # Metrics: split -> list[Metric]
    metrics: Dict[str, List[Metric]] = field(default_factory=dict)

    # Callbacks
    callbacks: List[Callback] = field(default_factory=list)

    # layer options
    layer_modes: dict[str, str] | None = None  # {"layername": "freeze"|"grad"|"meta"}
