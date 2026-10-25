import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "profile_train.py"


def test_profile_cli_writes_trace_and_table(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("profile_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = SCRIPT.parents[1]
    out = mod.main(["--config", str(root / "configs" / "train" / "debug-cpu.yaml"), "--out", str(tmp_path / "p"),
                    "--device", "cpu", "--steps", "1", "train.micro_batch_size=2", "train.global_batch_tokens=256",
                    "train.seq_len=128", "train.seq_len_warmup_tokens=0"])
    assert (out / "trace.json").stat().st_size > 0
    assert "aten::" in (out / "top_ops.txt").read_text(encoding="utf-8")
    assert "trace.json" in capsys.readouterr().out
