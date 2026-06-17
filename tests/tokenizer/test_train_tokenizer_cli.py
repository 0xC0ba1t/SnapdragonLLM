import importlib.util
import random
from pathlib import Path

from snapdragon.tokenizer.fast import FastTokenizer

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_tokenizer.py"


def _load():
    spec = importlib.util.spec_from_file_location("train_tokenizer_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_trains_and_saves(tmp_path):
    rng = random.Random(2)
    words = ["".join(rng.choice("abcdefgh") for _ in range(rng.randint(2, 6))) for _ in range(200)]
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(" ".join(rng.choice(words) for _ in range(10)) for _ in range(500)))
    out = tmp_path / "tok.json"
    _load().main(["--files", str(corpus), "--vocab-size", "300", "--max-gb", "0.001", "--out", str(out)])
    tok = FastTokenizer.from_file(out)
    assert tok.vocab_size == 300
