"""Periodic validation loss."""
from __future__ import annotations

import math

import torch
import torch.distributed as dist

from snapdragon.train.callbacks.base import Callback
from snapdragon.train.precision import autocast_context


@torch.no_grad()
def evaluate_loss(trainer, val_loader) -> float:
    """Mean cross-entropy (no z-loss) over the validation batches, averaged across ranks."""
    trainer.model.eval()
    total = torch.zeros(2, device=trainer.device)  # sum of per-batch CE, number of batches
    for batch in val_loader:
        batch = batch.to(trainer.device)
        with autocast_context(trainer.cfg.train.precision, trainer.device):
            _, ce, _ = trainer.model(batch.x, batch.y)
        total += torch.stack([ce.detach().float(), torch.ones((), device=trainer.device)])
    if trainer.world_size > 1 and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    trainer.model.train()
    return float(total[0] / total[1].clamp(min=1))


class ValLossCallback(Callback):
    """Adds ``val_loss`` and ``val_ppl`` (and ``val_loss_ema`` when a weight EMA is registered) to the
    metrics every ``every`` steps and on the last step."""

    priority = 30

    def __init__(self, val_loader, every: int) -> None:
        self.val_loader, self.every = val_loader, every

    def on_step_end(self, trainer, metrics: dict) -> None:
        step = metrics["step"]
        if step % self.every and step != trainer.total_steps:
            return
        loss = evaluate_loss(trainer, self.val_loader)
        metrics["val_loss"] = loss
        metrics["val_ppl"] = math.exp(min(loss, 20.0))
        trainer.state.best_val_loss = min(trainer.state.best_val_loss, loss)
        ema = trainer.extra_state.get("ema")
        if ema is not None:  # also score the averaged weights; they are usually the better model
            with ema.applied(trainer.raw_model):
                metrics["val_loss_ema"] = evaluate_loss(trainer, self.val_loader)
