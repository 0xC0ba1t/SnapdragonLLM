#!/usr/bin/env python3
"""Print the parameter breakdown, FLOPs per token and (optionally) a checkpoint's training state.

    python scripts/inspect_model.py --config configs/model/snapdragon-500m.yaml
    python scripts/inspect_model.py --checkpoint runs/x/checkpoints
"""
from __future__ import annotations

import argparse

from snapdragon.checkpoint.io import load_checkpoint
from snapdragon.checkpoint.model_io import resolve_checkpoint
from snapdragon.config.io import load_config
from snapdragon.model.factory import build_model, count_parameters
from snapdragon.model.flops import flops_per_token
from snapdragon.model.summary import format_summary


def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config")
    src.add_argument("--checkpoint")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    lines = []
    if args.checkpoint:
        path = resolve_checkpoint(args.checkpoint)
        meta = load_checkpoint(path, only=["meta", "trainer_state"])
        from snapdragon.config.loader import from_dict
        from snapdragon.config.schema import Config

        cfg = from_dict(Config, meta["meta"]["config"])
        st = meta["trainer_state"]
        lines.append(f"checkpoint {path.name}: step {st['step']}, {st['tokens_seen'] / 1e9:.3f}B tokens, "
                     f"{st['skipped_steps']} skipped steps, best val loss {st['best_val_loss']:.4f}")
    else:
        cfg = load_config(args.config, ["train.seq_len=64", "data.seq_len=64", *args.overrides])
    model = build_model(cfg, device="meta")
    lines.append(format_summary(model))
    m = cfg.model
    lines.append(f"{count_parameters(model) / 1e6:.1f}M parameters | d={m.d_model} L={m.n_layers} "
                 f"heads={m.n_heads}/{m.n_kv_heads} (q/kv) head_dim={m.head_dim} mlp={m.mlp_hidden} vocab={m.vocab_size}")
    lines.append(f"{flops_per_token(m, m.max_seq_len) / 1e9:.2f} GFLOPs/token at {m.max_seq_len} context "
                 f"(~{6 * count_parameters(model) / 1e9:.2f} by the 6N rule)")
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
