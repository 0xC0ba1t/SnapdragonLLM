import pytest
import torch

from snapdragon.data.packing import IGNORE_INDEX
from snapdragon.sft.data import SFTExample
from snapdragon.sft.packing import pack_examples

PAD = 0


def ex(ids, mask):
    return SFTExample(ids, mask)


def test_two_examples_share_a_row_with_restarting_positions():
    a = ex([1, 2, 3, 4], [0, 0, 1, 1])
    b = ex([5, 6, 7], [0, 1, 1])
    rows = pack_examples([a, b], seq_len=8, pad_id=PAD)
    assert rows.x.shape == (1, 8)
    assert rows.x[0].tolist() == [1, 2, 3, 4, 5, 6, 7, PAD]
    assert rows.position_ids[0].tolist() == [0, 1, 2, 3, 0, 1, 2, 0]


def test_targets_are_shifted_and_masked():
    a = ex([1, 2, 3, 4], [0, 0, 1, 1])
    rows = pack_examples([a], seq_len=6, pad_id=PAD)
    # inputs 1 2 3 4 pad pad -> targets (next token, only where mask) : ?, 3, 4, ?, ?, ?
    assert rows.y[0].tolist() == [IGNORE_INDEX, 3, 4, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]


def test_the_first_token_of_the_next_example_is_never_a_target():
    a = ex([1, 2, 3], [0, 1, 1])
    b = ex([7, 8, 9], [0, 1, 1])
    rows = pack_examples([a, b], seq_len=6, pad_id=PAD)
    # position 2 (token 3) would "predict" 7, the start of b: must be ignored
    assert rows.y[0, 2].item() == IGNORE_INDEX
    assert rows.y[0, 3].item() == 8  # inside b it is a normal target


def test_new_row_when_the_next_example_does_not_fit():
    a = ex([1, 2, 3, 4, 5], [0, 1, 1, 1, 1])
    b = ex([6, 7, 8, 9], [0, 1, 1, 1])
    rows = pack_examples([a, b], seq_len=6, pad_id=PAD)  # row width 7: 5 + 4 > 7
    assert rows.x.shape[0] == 2
    assert rows.x[1].tolist()[:4] == [6, 7, 8, 9]


def test_oversized_example_is_an_error():
    with pytest.raises(ValueError, match="does not fit"):
        pack_examples([ex(list(range(1, 10)), [1] * 9)], seq_len=4, pad_id=PAD)


def test_empty_input_gives_empty_tensors():
    rows = pack_examples([], seq_len=8, pad_id=PAD)
    assert rows.x.shape == (0, 8)
    assert rows.y.dtype == torch.long


def test_nothing_but_assistant_tokens_contribute_loss():
    a = ex([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 0])
    rows = pack_examples([a], seq_len=8, pad_id=PAD)
    live = (rows.y != IGNORE_INDEX).sum().item()
    assert live == 2
