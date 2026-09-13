import pytest
import torch

from snapdragon.eval.mc import MCExample, choice_losses, evaluate_mc, predict, render_mc


def test_render_masks_only_the_choice_tokens(small_tokenizer):
    ex = MCExample("abc def", ["ghi", "jab cde"], 0)
    tokens, mask = render_mc(small_tokenizer, ex)
    ctx_len = len(small_tokenizer.encode("abc def"))
    assert tokens.shape == mask.shape
    assert tokens.shape[0] == 2
    for i, choice in enumerate(ex.choices):
        n_ans = len(small_tokenizer.encode(choice))
        assert mask[i].sum().item() == n_ans
        assert mask[i, :ctx_len].sum().item() == 0


def test_rows_are_right_padded_with_pad_id(small_tokenizer):
    ex = MCExample("abc", ["a", "abc def ghi jab"], 1)
    tokens, mask = render_mc(small_tokenizer, ex)
    short_len = len(small_tokenizer.encode("abc")) + len(small_tokenizer.encode("a"))
    assert (tokens[0, short_len:] == small_tokenizer.pad_id).all()
    assert mask[0, short_len:].sum().item() == 0


def test_max_len_truncates_the_context_from_the_left(small_tokenizer):
    long_ctx = " ".join(["abc"] * 50)
    ex = MCExample(long_ctx, ["def"], 0)
    tokens, mask = render_mc(small_tokenizer, ex, max_len=10)
    assert tokens.shape[1] == 10
    assert mask[0].sum().item() == len(small_tokenizer.encode("def"))


def test_choice_losses_hand_computed():
    # vocab 4, one row of 3 tokens; position 0 predicts token 1, position 1 predicts token 2
    logits = torch.zeros(1, 3, 4)
    logits[0, 0, 2] = 5.0  # confident (and wrong) about the token after position 0
    tokens = torch.tensor([[0, 1, 3]])
    mask = torch.tensor([[0, 1, 1]])
    s, c = choice_losses(logits, tokens, mask)
    uniform = torch.log(torch.tensor(4.0))
    wrong = torch.logsumexp(logits[0, 0], 0) - logits[0, 0, 1]
    assert c.item() == 2
    assert s.item() == pytest.approx((wrong + uniform).item(), rel=1e-5)


def test_predict_normalisation_can_change_the_answer():
    sum_loss = torch.tensor([6.0, 8.0])  # choice 0 is better in total...
    count = torch.tensor([2.0, 8.0])  # ...but choice 1 is better per token
    assert predict(sum_loss, count) == (0, 1)


class _Oracle(torch.nn.Module):
    """Gives probability mass to a fixed 'correct' token string: scores choices by that token."""

    def __init__(self, vocab, good_token):
        super().__init__()
        self.vocab, self.good = vocab, good_token

    def forward(self, tokens):
        logits = torch.zeros(*tokens.shape, self.vocab)
        logits[..., self.good] = 10.0
        return logits


def test_evaluate_counts_correct_predictions(small_tokenizer):
    good = small_tokenizer.encode("a")[0]
    model = _Oracle(small_tokenizer.vocab_size, good)
    examples = [
        MCExample("ctx ", ["a", "bc"], 0),  # all-"a" choice is likelier
        MCExample("ctx ", ["bc", "a"], 1),
        MCExample("ctx ", ["a", "bc"], 1),  # wrong label on purpose
    ]
    out = evaluate_mc(model, small_tokenizer, examples)
    assert out["n"] == 3
    assert out["acc"] == pytest.approx(2 / 3)
    assert 0.0 <= out["acc_norm"] <= 1.0
