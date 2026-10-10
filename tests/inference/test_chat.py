import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.inference.chat import ChatSession
from snapdragon.inference.sampling import SamplingConfig
from snapdragon.model.factory import build_model
from snapdragon.tokenizer.chat import encode_chat


@pytest.fixture(scope="module")
def lm(small_tokenizer):
    cfg = ModelConfig(vocab_size=small_tokenizer.vocab_size, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                      mlp_hidden=64, max_seq_len=96)
    torch.manual_seed(0)
    return build_model(cfg).eval()


def _session(lm, tok, **kw):
    return ChatSession(lm, tok, sampling=SamplingConfig(temperature=0), max_new_tokens=8, **kw)


def test_history_records_both_sides(lm, small_tokenizer):
    s = _session(lm, small_tokenizer, system="be brief")
    reply = s.reply("abc def")
    assert [m["role"] for m in s.messages] == ["system", "user", "assistant"]
    assert s.messages[-1]["content"] == reply


def test_reply_never_contains_special_markers(lm, small_tokenizer):
    out = _session(lm, small_tokenizer).reply("abc def ghi")
    assert "<|end_turn|>" not in out
    assert "<|eos|>" not in out


def test_second_turn_prompt_contains_the_first_exchange(lm, small_tokenizer):
    s = _session(lm, small_tokenizer)
    s.reply("abc def")
    s.messages.append({"role": "user", "content": "ghi jab"})
    ids = s._prompt_ids()
    expected, _ = encode_chat(small_tokenizer, s.messages, add_generation_prompt=True)
    assert ids == expected


def test_old_turns_are_dropped_when_the_context_overflows_but_system_is_kept(lm, small_tokenizer):
    s = _session(lm, small_tokenizer, system="sys")
    for _ in range(30):
        s.messages.append({"role": "user", "content": "abc def ghi jab " * 3})
        s.messages.append({"role": "assistant", "content": "cde fgh " * 3})
    s.messages.append({"role": "user", "content": "last question"})
    ids = s._prompt_ids()
    assert len(ids) <= lm.cfg.max_seq_len - s.max_new_tokens
    text = small_tokenizer.decode(ids)
    assert "sys" in text
    assert "last question" in text


def test_a_single_oversized_message_is_truncated_from_the_left(lm, small_tokenizer):
    s = _session(lm, small_tokenizer)
    s.messages.append({"role": "user", "content": "abc " * 500})
    ids = s._prompt_ids()
    assert len(ids) == lm.cfg.max_seq_len - s.max_new_tokens
    assert ids[0] == small_tokenizer.bos_id


def test_reset_keeps_only_the_system_message(lm, small_tokenizer):
    s = _session(lm, small_tokenizer, system="sys")
    s.reply("abc")
    s.reset()
    assert s.messages == [{"role": "system", "content": "sys"}]


def test_streaming_chunks_join_to_the_final_reply(lm, small_tokenizer):
    s = _session(lm, small_tokenizer)
    chunks = list(s.reply_stream("abc def", seed=1))
    assert "".join(chunks) == s.messages[-1]["content"]
