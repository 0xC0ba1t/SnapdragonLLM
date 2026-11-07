#!/usr/bin/env python3
"""Regenerate docs/configuration.md from the config dataclasses.

    python scripts/gen_config_docs.py            # write the file
    python scripts/gen_config_docs.py --check    # exit 1 if it is out of date (used in CI)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from snapdragon.config.docs import render_config_reference

TARGET = Path(__file__).resolve().parents[1] / "docs" / "configuration.md"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = render_config_reference()
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print("docs/configuration.md is out of date; run scripts/gen_config_docs.py")
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
