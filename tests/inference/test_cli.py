import importlib.util
import io
from pathlib import Path

import pytest

from snapdragon.train.checkpointing import checkpoint_root, save_trainer_checkpoint

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_cli", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def assets(make_trainer, tiny_config, tmp_path, small_tokenizer):
    cfg = tiny_config()
    cfg.model.vocab_size = small_tokenizer.vocab_size
    cfg.model.max_seq_len = cfg.train.seq_len = cfg.data.seq_len = 64
    cfg.train.global_batch_tokens = 2 * 64 * 2
    cfg.train.max_tokens = cfg.train.global_batch_tokens * 5
    cfg.train.out_dir = str(tmp_path)
    t = make_trainer(cfg)
    t.train_step()
    save_trainer_checkpoint(t)
    small_tokenizer.save(tmp_path / "tok.json")
    return ["--checkpoint", str(checkpoint_root(t)), "--tokenizer", str(tmp_path / "tok.json"), "--device", "cpu"]


def test_generate_cli_streams_and_is_seed_reproducible(assets):
    gen = _load("generate")
    args = [*assets, "--prompt", "abc def", "--max-new-tokens", "10", "--seed", "3", "--temperature", "1.0"]
    buf1, buf2 = io.StringIO(), io.StringIO()
    a = gen.main(args, out=buf1)
    b = gen.main(args, out=buf2)
    assert a == b
    assert buf1.getvalue().endswith("\n")
    assert buf1.getvalue().strip() == a.strip()


def test_generate_cli_greedy_is_deterministic_without_a_seed(assets):
    gen = _load("generate")
    args = [*assets, "--prompt", "abc", "--max-new-tokens", "6", "--temperature", "0"]
    assert gen.main(args, out=io.StringIO()) == gen.main(args, out=io.StringIO())


def test_chat_cli_loop_commands(assets):
    chat = _load("chat")
    inputs = iter(["abc def", "", "/reset", "ghi", "/quit"])
    shown = []
    session = chat.main([*assets, "--max-new-tokens", "6", "--temperature", "0"],
                        read=lambda prompt: next(inputs), write=shown.append)
    assert any(s.startswith("bot> ") for s in shown)
    assert "(conversation cleared)" in shown
    assert [m["role"] for m in session.messages] == ["user", "assistant"]  # only the post-reset exchange


def test_chat_cli_exits_on_eof(assets):
    chat = _load("chat")

    def eof(prompt):
        raise EOFError

    session = chat.main([*assets], read=eof, write=lambda s: None)
    assert session.messages == []
