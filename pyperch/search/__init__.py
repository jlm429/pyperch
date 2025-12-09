
"""
Search utilities for hyperparameter optimization.

This package provides optional integration points for external search
libraries such as Optuna, without depending on the core training logic
of the library.
"""

from .strategy import SearchStrategy, OptunaStrategy
from .builder import TrainConfigBuilder
from .adapter import TrainerAdapter
from .optuna_search import OptunaSearch
