import json
import logging

from snapdragon.train.callbacks.loggers import ConsoleLogger, JsonlLogger, format_step


class FakeTrainer:
    is_main = True
    total_steps = 100


def test_format_includes_optional_fields_only_when_present():
    line = format_step({"step": 5, "loss": 2.5, "lr": 1e-3, "grad_norm": 0.7}, 100)
    assert "step      5/100" in line
    assert "loss 2.5000" in line
    assert "tok/s" not in line
    full = format_step({"step": 5, "loss": 2.5, "tokens_per_sec": 12_000, "mfu": 0.31,
                        "mem_peak_gb": 9.2, "skipped": True}, 100)
    assert "12.0k tok/s" in full
    assert "mfu 31.0%" in full
    assert "mem 9.2GB" in full
    assert "SKIPPED" in full


def test_console_logs_every_n_and_on_last_step(caplog):
    cb = ConsoleLogger(every=10)
    with caplog.at_level(logging.INFO, logger="train"):
        for step in (5, 10, 15, 100):
            cb.on_step_end(FakeTrainer(), {"step": step, "loss": 1.0})
    assert len(caplog.records) == 2  # step 10 and the final step 100


def test_console_silent_on_non_main_rank(caplog):
    class Worker(FakeTrainer):
        is_main = False

    with caplog.at_level(logging.INFO, logger="train"):
        ConsoleLogger(every=1).on_step_end(Worker(), {"step": 1, "loss": 1.0})
    assert caplog.records == []


def test_jsonl_appends_valid_json_even_for_nan(tmp_path):
    cb = JsonlLogger(tmp_path / "run", every=1)
    cb.on_train_start(FakeTrainer())
    cb.on_step_end(FakeTrainer(), {"step": 1, "loss": 2.0})
    cb.on_step_end(FakeTrainer(), {"step": 2, "loss": float("nan")})
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[1]["loss"] is None


def test_jsonl_respects_every(tmp_path):
    cb = JsonlLogger(tmp_path, every=5)
    cb.on_train_start(FakeTrainer())
    for s in range(1, 11):
        cb.on_step_end(FakeTrainer(), {"step": s, "loss": 1.0})
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
