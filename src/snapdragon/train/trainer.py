"""The training loop."""
from __future__ import annotations

import contextlib
import math

import torch
import torch.distributed as dist
from torch import nn

from snapdragon.config.schema import Config
from snapdragon.model.compile import unwrap_model
from snapdragon.optim.clip import clip_grad_norm
from snapdragon.train.callbacks.base import Callback, CallbackList
from snapdragon.train.curriculum import seq_len_at
from snapdragon.train.precision import autocast_context
from snapdragon.train.state import TrainState


def to_host(*tensors: torch.Tensor) -> list[float]:
    """Fetch several scalar tensors with ONE device-to-host copy (each ``float(t)`` on a CUDA tensor is
    a separate synchronisation that stalls the CPU until the GPU catches up)."""
    flat = torch.cat([t.detach().reshape(-1).float() for t in tensors])
    return flat.tolist()


class Trainer:
    """Runs optimizer steps; everything optional (logging, checkpoints, eval) is a callback.

    ``train_loader`` must yield batches with ``.x``/``.y``/``.position_ids`` and expose
    ``state_dict()``. One *step* = ``grad_accum_steps`` micro-batches per rank = one optimizer update.
    """

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        optimizer,
        schedule,
        train_loader,
        device: torch.device | str = "cpu",
        callbacks: list[Callback] | None = None,
        rank: int = 0,
        world_size: int = 1,
        state: TrainState | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.raw_model = unwrap_model(model)
        self.optimizer = optimizer
        self.schedule = schedule
        self.train_loader = train_loader
        self.device = torch.device(device)
        self.rank, self.world_size = rank, world_size
        self.state = state or TrainState()
        self.callbacks = CallbackList(callbacks or [])
        self.grad_accum_steps = cfg.train.grad_accum_steps(world_size)
        self.total_steps = cfg.train.total_steps
        self.stop_requested = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        """Mean across ranks (SUM then divide: gloo has no AVG)."""
        if self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values = values / self.world_size
        return values

    def train_step(self) -> dict:
        """One optimizer update. Returns the step's metrics."""
        cfg = self.cfg.train
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        stats = torch.zeros(3, device=self.device)  # loss, ce, z averaged over micro-batches
        tokens = 0
        seq_len = seq_len_at(self.state.tokens_seen, cfg)  # curriculum: shorter rows early on
        for micro in range(self.grad_accum_steps):
            batch = next(self.train_loader).to(self.device).regroup(seq_len)
            sync = contextlib.nullcontext()
            if micro < self.grad_accum_steps - 1 and hasattr(self.model, "no_sync"):
                sync = self.model.no_sync()  # DDP: only all-reduce gradients on the last micro-batch
            with sync, autocast_context(cfg.precision, self.device):
                loss, ce, z = self.model(batch.x, batch.y, batch.position_ids)
            (loss / self.grad_accum_steps).backward()
            stats += torch.stack([loss.detach(), ce.detach(), z.detach()]) / self.grad_accum_steps
            tokens += batch.x.numel()
        stats = self._reduce(stats)
        norm = clip_grad_norm(self.raw_model.parameters(), self.cfg.optim.grad_clip)

        loss_val, ce_val, z_val, norm_val = to_host(stats, norm)
        finite = math.isfinite(loss_val) and math.isfinite(norm_val)
        spike = (
            cfg.loss_spike_factor > 0
            and self.state.loss_ema > 0
            and loss_val > cfg.loss_spike_factor * self.state.loss_ema
        )
        skipped = not finite or spike
        lr = self.optimizer.param_groups[0]["lr"]  # the LR this step actually used
        if skipped:
            self.optimizer.zero_grad(set_to_none=True)
            self.state.skipped_steps += 1
        else:
            self.optimizer.step()
            self.state.update_loss_ema(loss_val)
        self.schedule.step()
        self.state.step += 1
        self.state.tokens_seen += tokens * self.world_size
        return {
            "step": self.state.step,
            "loss": loss_val,
            "ce": ce_val,
            "z": z_val,
            "grad_norm": norm_val,
            "lr": lr,
            "tokens_seen": self.state.tokens_seen,
            "seq_len": seq_len,
            "skipped": skipped,
        }

    def fit(self) -> TrainState:
        self.callbacks.on_train_start(self)
        try:
            while self.state.step < self.total_steps and not self.stop_requested:
                metrics = self.train_step()
                self.callbacks.on_step_end(self, metrics)
        except BaseException as exc:
            self.callbacks.on_exception(self, exc)
            raise
        self.callbacks.on_train_end(self)
        return self.state
