#!/usr/bin/env python3
"""Interactive chat with a checkpoint (use an SFT-tuned model for sensible behaviour).

    python scripts/chat.py --checkpoint runs/sft/checkpoints --tokenizer data/tokenizer/snapdragon-32k.json

Commands: /reset clears the conversation, /quit (or Ctrl-D) exits.
"""
from __future__ import annotations

import argparse

from snapdragon.checkpoint.model_io import load_model_from_checkpoint
from snapdragon.inference.chat import ChatSession
from snapdragon.inference.sampling import SamplingConfig
from snapdragon.tokenizer.fast import FastTokenizer
from snapdragon.utils.device import get_device


def main(argv: list[str] | None = None, read=input, write=print) -> ChatSession:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    model, _ = load_model_from_checkpoint(args.checkpoint, get_device(args.device))
    tok = FastTokenizer.from_file(args.tokenizer)
    session = ChatSession(model, tok, args.system, SamplingConfig(args.temperature, top_p=args.top_p),
                          args.max_new_tokens)
    while True:
        try:
            line = read("you> ").strip()
        except EOFError:
            break
        if line in ("/quit", "/exit"):
            break
        if line == "/reset":
            session.reset()
            write("(conversation cleared)")
            continue
        if not line:
            continue
        write("bot> " + session.reply(line))
    return session


if __name__ == "__main__":
    main()
