"""AdamW with both moments stored as blockwise int8 codes (2 bytes/param instead of 8)."""
from __future__ import annotations

import torch

from snapdragon.optim.quant import (
    DEFAULT_BLOCK,
    dequantize_blockwise,
    power_codebook,
    quantize_blockwise,
)


class AdamW8bit(torch.optim.Optimizer):
    """Decoupled-weight-decay Adam. Parameters with fewer than ``min_8bit_size`` elements keep exact
    fp32 state (tiny vectors like norm gains gain nothing from quantization and are sensitive).

    The first moment uses a signed power codebook; the second moment is stored as sqrt(v) on an
    unsigned power codebook, so the quantity that divides the update keeps good relative precision.
    """

    def __init__(
        self,
        params,
        lr: float = 6e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        block: int = DEFAULT_BLOCK,
        min_8bit_size: int = 4096,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        block=block, min_8bit_size=min_8bit_size)
        super().__init__(params, defaults)
        self.m_codebook = power_codebook(True, 3.0)
        self.sqrt_v_codebook = power_codebook(False, 2.0)

    def _init_state(self, p: torch.Tensor, group: dict) -> dict:
        state = {"step": 0}
        if p.numel() >= group["min_8bit_size"]:
            n_blocks = -(-p.numel() // group["block"])
            zeros8 = torch.zeros(p.numel(), dtype=torch.uint8, device=p.device)
            state.update(
                m_codes=zeros8.clone() + self._zero_code(self.m_codebook),
                m_absmax=torch.zeros(n_blocks, device=p.device),
                v_codes=zeros8,
                v_absmax=torch.zeros(n_blocks, device=p.device),
            )
        else:
            state.update(exp_avg=torch.zeros_like(p, dtype=torch.float32),
                         exp_avg_sq=torch.zeros_like(p, dtype=torch.float32))
        return state

    @staticmethod
    def _zero_code(codebook: torch.Tensor) -> int:
        return int((codebook - 0).abs().argmin())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd, block = (group[k] for k in ("lr", "betas", "eps", "weight_decay", "block"))
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if not state:
                    state.update(self._init_state(p, group))
                state["step"] += 1
                quantized = "m_codes" in state
                if quantized:
                    m = dequantize_blockwise(state["m_codes"], state["m_absmax"], self.m_codebook, p.shape, block)
                    sqrt_v = dequantize_blockwise(state["v_codes"], state["v_absmax"], self.sqrt_v_codebook, p.shape, block)
                    v = sqrt_v.square()
                else:
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                m = m.mul(b1).add_(g, alpha=1 - b1)
                v = v.mul(b2).addcmul_(g, g, value=1 - b2)
                bc1 = 1 - b1 ** state["step"]
                bc2 = 1 - b2 ** state["step"]
                denom = (v / bc2).sqrt_().add_(eps)
                if wd:
                    p.mul_(1 - lr * wd)
                p.addcdiv_((m / bc1).to(p.dtype), denom.to(p.dtype), value=-lr)
                if quantized:
                    state["m_codes"], state["m_absmax"] = quantize_blockwise(m, self.m_codebook, block)
                    state["v_codes"], state["v_absmax"] = quantize_blockwise(v.sqrt(), self.sqrt_v_codebook, block)
                else:
                    state["exp_avg"], state["exp_avg_sq"] = m, v
        return loss
