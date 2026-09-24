"""Run benchmark tasks during training (HellaSwag etc.), the way v2 did every 250 steps."""
from __future__ import annotations

from snapdragon.eval.runner import run_tasks
from snapdragon.tokenizer.fast import FastTokenizer
from snapdragon.train.callbacks.base import Callback
from snapdragon.train.precision import autocast_context


class BenchmarkCallback(Callback):
    """Adds ``<task>_acc`` / ``<task>_acc_norm`` (or ``lambada_ppl``) to the metrics every ``every``
    steps and on the last step. Runs on rank 0 only; other ranks skip, so evaluation costs no extra
    synchronisation (the model is evaluated un-wrapped, with its own weights)."""

    priority = 35

    def __init__(self, tokenizer: FastTokenizer, tasks: list[str], every: int, limit: int = 0,
                 task_kwargs: dict[str, dict] | None = None) -> None:
        self.tokenizer, self.tasks, self.every, self.limit = tokenizer, tasks, every, limit
        self.task_kwargs = task_kwargs or {}

    def on_step_end(self, trainer, metrics: dict) -> None:
        step = metrics["step"]
        if not trainer.is_main or (step % self.every and step != trainer.total_steps):
            return
        results = run_tasks(
            self.tasks, trainer.raw_model, self.tokenizer, trainer.device, self.limit,
            max_len=trainer.raw_model.cfg.max_seq_len,
            autocast=autocast_context(trainer.cfg.train.precision, trainer.device),
            task_kwargs=self.task_kwargs,
        )
        trainer.raw_model.train()
        for task, res in results.items():
            for key in ("acc", "acc_norm", "ppl"):
                if key in res:
                    metrics[f"{task}_{key}"] = res[key]
