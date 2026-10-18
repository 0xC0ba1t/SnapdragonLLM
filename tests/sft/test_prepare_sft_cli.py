import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_sft.py"


def _cli():
    spec = importlib.util.spec_from_file_location("prepare_sft_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_converts_filters_and_writes_unicode_jsonl(tmp_path):
    src = tmp_path / "in.jsonl"
    rows = [{"instruction": f"q{i}", "input": "", "output": f"café {i}"} for i in range(5)]
    rows.append({"instruction": "bad", "input": "", "output": ""})
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "sub" / "out.jsonl"
    n = _cli().main(["--files", str(src), "--format", "alpaca", "--out", str(out)])
    assert n == 5
    lines = out.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["messages"][1]["content"] == "café 0"
    assert "café" in lines[0]  # ensure_ascii=False keeps the text readable


def test_shuffle_is_seeded(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps({"instruction": f"q{i}", "output": "a"}) for i in range(20)), encoding="utf-8")
    cli = _cli()
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    cli.main(["--files", str(src), "--format", "alpaca", "--out", str(a), "--shuffle-seed", "4"])
    cli.main(["--files", str(src), "--format", "alpaca", "--out", str(b), "--shuffle-seed", "4"])
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    plain = tmp_path / "c.jsonl"
    cli.main(["--files", str(src), "--format", "alpaca", "--out", str(plain)])
    assert plain.read_text(encoding="utf-8") != a.read_text(encoding="utf-8")
