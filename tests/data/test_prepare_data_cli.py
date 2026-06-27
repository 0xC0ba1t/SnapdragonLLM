import importlib.util
from pathlib import Path

from snapdragon.data.manifest import DatasetManifest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_data.py"


def _load():
    spec = importlib.util.spec_from_file_location("prepare_data_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_end_to_end_on_local_files(small_tokenizer, tmp_path):
    tok_path = tmp_path / "tok.json"
    small_tokenizer.save(tok_path)
    text = tmp_path / "docs.txt"
    text.write_text("abc defg hij klm nop qrs tuv " * 40)
    out = tmp_path / "ds"
    cli = _load()
    cli.main(["--spec", f"files:{text}", "--tokenizer", str(tok_path), "--out", str(out),
              "--val-tokens", "0", "--no-filter", "--shard-tokens", "100000"])
    m = DatasetManifest.load(out)
    assert m.total_tokens("train") > 0
    cli.main(["--spec", "files:x", "--tokenizer", str(tok_path), "--out", str(out), "--verify"])
