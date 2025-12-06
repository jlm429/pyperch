from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def freeze_layers(model: nn.Module, prefixes: Iterable[str]) -> None:
    prefixes = tuple(prefixes)
    for name, param in model.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad = False
