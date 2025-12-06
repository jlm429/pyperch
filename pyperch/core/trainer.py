from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config.schema import TrainConfig
from ..optim import run_optimizer_step
from ..utils.seed import set_seed
from .engine import Engine
from .callbacks import CallbackList, HistoryCallback
from .metrics import Metric


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        config: TrainConfig,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config

        # RNG / device setup
        if config.seed is not None:
            set_seed(config.seed)
            self.rng = np.random.default_rng(config.seed)
        else:
            self.rng = np.random.default_rng()

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # ------------------------------------------------------------
        # Partition parameters into freeze / grad / meta groups
        # ------------------------------------------------------------
        self.grad_params: list[torch.nn.Parameter] = []
        self.meta_params: list[torch.nn.Parameter] = []
        self.frozen_params: list[torch.nn.Parameter] = []

        layer_modes = self.config.layer_modes or {}

        for name, p in self.model.named_parameters():
            mode = layer_modes.get(name, "meta")  # default is meta-optimizer

            if mode == "freeze":
                p.requires_grad = False
                self.frozen_params.append(p)

            elif mode == "grad":
                p.requires_grad = True
                self.grad_params.append(p)

            elif mode == "meta":
                p.requires_grad = True
                self.meta_params.append(p)

            else:
                raise ValueError(f"Unknown layer mode '{mode}' for parameter '{name}'")

        # Torch optimizer for gradient-based params (if any)
        if self.grad_params:
            # TODO: expose LR / optimizer choice in TrainConfig
            self.torch_optim = torch.optim.Adam(self.grad_params, lr=1e-3)
        else:
            self.torch_optim = None

        # Callbacks
        user_cbs = list(config.callbacks)
        self.history_cb = HistoryCallback()
        user_cbs.append(self.history_cb)
        self.callbacks = CallbackList(user_cbs)

        # Metrics and engine
        self.metrics: Dict[str, List[Metric]] = config.metrics
        self.engine = Engine(model=self.model, config=config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, train_loader: DataLoader, valid_loader: DataLoader | None = None):
        self.callbacks.on_train_begin(self.engine)

        if self.config.optimizer_mode not in {"per_epoch", "per_batch"}:
            raise ValueError("optimizer_mode must be 'per_epoch' or 'per_batch'")

        for epoch in range(self.config.max_epochs):
            self.engine.epoch = epoch
            self.engine.step = 0
            self.callbacks.on_epoch_begin(self.engine)

            if self.config.optimizer_mode == "per_epoch":
                train_loss = self._run_epoch_per_epoch_mode(train_loader)
            else:
                train_loss = self._run_epoch_per_batch_mode(train_loader)

            self.engine.train_loss = train_loss

            # Evaluate for logging
            train_loss_eval, train_metrics = self._evaluate_split(train_loader, "train")
            valid_loss_eval, valid_metrics = (None, {})
            if valid_loader is not None:
                valid_loss_eval, valid_metrics = self._evaluate_split(valid_loader, "valid")  # type: ignore

            self.engine.train_loss = (
                float(train_loss_eval) if train_loss_eval is not None else None
            )
            self.engine.valid_loss = (
                float(valid_loss_eval) if valid_loss_eval is not None else None
            )
            if train_metrics:
                self.engine.set_metrics("train", train_metrics)
            if valid_metrics:
                self.engine.set_metrics("valid", valid_metrics)

            self.callbacks.on_epoch_end(self.engine)

            if self.engine.stop_training:
                break

        self.callbacks.on_train_end(self.engine)
        return self.history_cb.history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _closure_full(self, loader: DataLoader) -> float:
        """Deterministic, full-dataset loss (ABAGAIL style)."""
        self.model.eval()  # critical for SA!
        total_loss = 0.0
        n = 0

        with torch.no_grad():  # critical for SA!
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = self.model(x)
                loss = self.loss_fn(preds, y)
                total_loss += float(loss)
                n += 1

        return total_loss / max(n, 1)

    def _run_epoch_per_epoch_mode(self, train_loader: DataLoader) -> float:
        """Run ONE SA/RHC/GA step per epoch using full-dataset loss."""
        params = self.meta_params

        if not params:
            return self._closure_full(train_loader)

        # deterministic closure with eval() and no_grad()
        def closure() -> float:
            return self._closure_full(train_loader)

        self.model.eval()  # matches old optimizer environment

        loss = run_optimizer_step(
            name=self.config.optimizer,
            params=params,
            rng=self.rng,
            closure=closure,
            cfg=self.config.optimizer_config,
        )

        return float(loss)

    def _run_epoch_per_batch_mode(self, train_loader: DataLoader) -> float:
        """Per-batch loop with optional grad + meta updates."""
        params_meta = self.meta_params
        running_loss = 0.0
        n = 0

        for batch in train_loader:
            self.engine.step += 1
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            base_loss_val: float

            # ---------------------------------------
            # Gradient optimizer step (if any)
            # ---------------------------------------
            if self.torch_optim:
                self.torch_optim.zero_grad()
                preds = self.model(x)
                grad_loss = self.loss_fn(preds, y)
                grad_loss.backward()
                self.torch_optim.step()
                base_loss_val = float(grad_loss.detach().cpu())
            else:
                # If no grad optimizer, compute a loss value for logging
                self.model.train()
                with torch.no_grad():
                    preds = self.model(x)
                    loss = self.loss_fn(preds, y)
                    base_loss_val = float(loss.detach().cpu())

            # ---------------------------------------
            # Closure used for meta optimization
            # ---------------------------------------
            def closure() -> float:
                self.model.train()
                with torch.no_grad():
                    preds = self.model(x)
                    loss = self.loss_fn(preds, y)
                    return float(loss.detach().cpu())

            # ---------------------------------------
            # Meta optimizer (RHC/SA/GA)
            # ---------------------------------------
            if params_meta:
                meta_loss = run_optimizer_step(
                    name=self.config.optimizer,
                    params=params_meta,
                    rng=self.rng,
                    closure=closure,
                    cfg=self.config.optimizer_config,
                )
                running_loss += float(meta_loss)
            else:
                running_loss += base_loss_val

            n += 1
            self.callbacks.on_step_end(self.engine)

        return running_loss / max(n, 1)

    def _evaluate_split(
        self,
        loader: DataLoader,
        split: str,
    ) -> Tuple[float | None, Dict[str, float]]:
        metric_list = self.metrics.get(split)
        if metric_list:
            for m in metric_list:
                m.reset()

        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = self.model(x)
                loss = self.loss_fn(preds, y)
                total_loss += float(loss.detach().cpu())
                n += 1

                if metric_list:
                    for m in metric_list:
                        m.update(preds, y)

        if n == 0:
            return None, {}

        metrics_out: Dict[str, float] = {}
        if metric_list:
            for m in metric_list:
                metrics_out[m.name] = float(m.compute())

        return total_loss / max(n, 1), metrics_out
