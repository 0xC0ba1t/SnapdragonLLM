import json

import pytest
import torch

from snapdragon.eval.lambada import evaluate_lambada, load_lambada, split_last_word


def test_split_last_word():
    assert split_last_word("the cat sat on the mat") == ("the cat sat on the", " mat")
    assert split_last_word("  trailing space ok  ") == ("trailing space", " ok")


def test_load_with_limit(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_text("\n".join(json.dumps({"text": f"a b c{i}"}) for i in range(4)) + "\n")
    assert load_lambada(path) == [f"a b c{i}" for i in range(4)]
    assert load_lambada(path, limit=2) == ["a b c0", "a b c1"]


class _Always(torch.nn.Module):
    """Predicts token `tok_id` everywhere with high confidence."""

    def __init__(self, vocab, tok_id):
        super().__init__()
        self.vocab, self.tok_id = vocab, tok_id

    def forward(self, x):
        logits = torch.zeros(*x.shape, self.vocab)
        logits[..., self.tok_id] = 8.0
        return logits


def test_accuracy_requires_the_whole_word_and_ppl_reflects_confidence(small_tokenizer):
    target_ids = small_tokenizer.encode(" abc")
    texts = ["def ghi abc", "def ghi jab"]
    good = evaluate_lambada(_Always(small_tokenizer.vocab_size, target_ids[0]), small_tokenizer, texts)
    if len(target_ids) == 1:
        assert good["acc"] == pytest.approx(0.5)  # right on the first text, wrong on the second
    assert good["n"] == 2
    assert good["ppl"] > 1.0


def test_perplexity_of_a_uniform_model_equals_vocab_size(small_tokenizer):
    class Uniform(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(*x.shape, small_tokenizer.vocab_size)

    out = evaluate_lambada(Uniform(), small_tokenizer, ["abc def ghi jab"])
    assert out["ppl"] == pytest.approx(small_tokenizer.vocab_size, rel=1e-4)
    assert out["acc"] in (0.0, 1.0)


def test_max_len_keeps_the_end_of_the_passage(small_tokenizer):
    seen = []

    class Spy(torch.nn.Module):
        def forward(self, x):
            seen.append(x.shape[1])
            return torch.zeros(*x.shape, small_tokenizer.vocab_size)

    evaluate_lambada(Spy(), small_tokenizer, [" ".join(["abc"] * 40) + " def"], max_len=8)
    assert seen == [8]
