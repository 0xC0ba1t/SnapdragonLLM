"""Training loop and evaluation configuration."""
from __future__ import annotations

from dataclasses import dataclass, field

# fp16 would need a GradScaler, which the custom Muon/8-bit optimizers do not support
PRECISIONS = ("bf16", "fp32")


@dataclass
class EvalConfig:
    tasks: list[str] = field(default_factory=lambda: ["hellaswag"])
    every_steps: int = 1000
    limit: int = 0  # 0 means the full task
    val_batches: int = 20


@dataclass
class TrainConfig:
    run_name: str = "snapdragon-500m"
    out_dir: str = "runs"
    seed: int = 1337
    global_batch_tokens: int = 524_288
    micro_batch_size: int = 4
    seq_len: int = 2048
    max_tokens: int = 10_000_000_000
    precision: str = "bf16"
    compile: str = "auto"  # auto | true | false
    seq_len_start: int = 512  # sequence length curriculum: grows to seq_len ...
    seq_len_warmup_tokens: int = 0  # ... over this many tokens (0 disables)
    log_every: int = 10
    save_every: int = 500
    keep_last_n: int = 3
    resume: str = "auto"  # auto | none | path to a checkpoint
    loss_spike_factor: float = 0.0  # skip steps whose loss exceeds factor times the EMA (0 disables)
    z_loss_coeff: float = 1e-4
    ce_chunk_tokens: int = 2048
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        if self.compile not in ("auto", "true", "false"):
            raise ValueError("compile must be auto, true or false")
        if self.micro_batch_size < 1 or self.seq_len < 2:
            raise ValueError("micro_batch_size and seq_len must be positive")
        if self.seq_len_warmup_tokens and not 2 <= self.seq_len_start <= self.seq_len:
            raise ValueError("seq_len_start must be between 2 and seq_len")

    def grad_accum_steps(self, world_size: int = 1) -> int:
        per_step = self.micro_batch_size * self.seq_len * world_size
        if self.global_batch_tokens % per_step:
            raise ValueError(
                f"global_batch_tokens {self.global_batch_tokens} is not divisible by "
                f"micro_batch_size*seq_len*world_size = {per_step}"
            )
        return self.global_batch_tokens // per_step

    @property
    def total_steps(self) -> int:
        return self.max_tokens // self.global_batch_tokens
