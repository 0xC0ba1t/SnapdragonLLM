import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.inference.sampling import SamplingConfig
from snapdragon.inference.text import Detokenizer, StopStringFilter, stream_text
from snapdragon.model.factory import build_model


def test_detokenizer_reassembles_text_exactly(small_tokenizer):
    text = "abc def ghi jab cde fgh ijk"
    d = Detokenizer(small_tokenizer)
    out = "".join(d.push(i) for i in small_tokenizer.encode(text)) + d.flush()
    assert out == text


def test_multibyte_characters_are_never_split(small_tokenizer):
    text = "abc 你好 \U0001f642 def"  # the tiny vocab has no merges for these: many byte tokens
    ids = small_tokenizer.encode(text)
    d = Detokenizer(small_tokenizer)
    pieces = [d.push(i) for i in ids]
    assert "�" not in "".join(pieces)
    assert "".join(pieces) + d.flush() == text
    assert any(p == "" for p in pieces)  # it did hold something back mid-character


def test_flush_emits_a_dangling_partial_character_as_replacement(small_tokenizer):
    ids = small_tokenizer.encode("你")
    d = Detokenizer(small_tokenizer)
    assert d.push(ids[0]) == ""
    assert "�" in d.flush()


def test_stop_filter_cuts_at_the_stop_and_holds_back_prefixes():
    f = StopStringFilter(["END"])
    out = f.feed("hello E") + f.feed("N") + f.feed("D and more")
    assert out == "hello "
    assert f.done
    assert f.feed("ignored") == ""


def test_stop_filter_releases_held_text_when_it_is_not_a_stop():
    f = StopStringFilter(["END"])
    out = f.feed("abc E") + f.feed("Nx") + f.flush()
    assert out == "abc ENx"
    assert not f.done


def test_stop_filter_with_multiple_stops_cuts_at_the_earliest():
    f = StopStringFilter(["bb", "a"])
    assert f.feed("xxbbxa") == "xx"
    assert f.done


def test_no_stops_passes_everything_through():
    f = StopStringFilter([])
    assert f.feed("hello") == "hello"
    assert f.flush() == ""


@pytest.fixture(scope="module")
def lm(small_tokenizer):
    cfg = ModelConfig(vocab_size=small_tokenizer.vocab_size, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                      mlp_hidden=64, max_seq_len=64)
    torch.manual_seed(0)
    return build_model(cfg).eval()


def test_stream_text_matches_decoding_the_generated_ids(lm, small_tokenizer):
    from snapdragon.inference.generate import generate

    cfg = SamplingConfig(temperature=0)
    ids = [small_tokenizer.bos_id, *small_tokenizer.encode("abc def")]
    reference = generate(lm, ids, 12, cfg, stop_ids=[small_tokenizer.eos_id]).token_ids
    if reference and reference[-1] == small_tokenizer.eos_id:
        reference = reference[:-1]
    text = "".join(stream_text(lm, small_tokenizer, "abc def", 12, cfg))
    assert text == small_tokenizer.decode(reference)


def test_stream_text_stops_at_a_stop_string(lm, small_tokenizer):
    cfg = SamplingConfig(temperature=0)
    full = "".join(stream_text(lm, small_tokenizer, "abc def", 20, cfg))
    if len(full) < 4:
        pytest.skip("generated text too short for this check")
    stop = full[2:4]
    cut = "".join(stream_text(lm, small_tokenizer, "abc def", 20, cfg, stop_strings=[stop]))
    assert cut == full[: full.find(stop)]
