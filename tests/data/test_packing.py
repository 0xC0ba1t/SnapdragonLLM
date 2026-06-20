import pytest
import torch

from snapdragon.data.packing import doc_ids_from_tokens, position_ids_from_tokens, split_window

EOS = 9


def _x(*rows):
    return torch.tensor(rows)


def test_doc_ids():
    x = _x([1, 2, EOS, 3, 4, EOS, 5])
    assert doc_ids_from_tokens(x, EOS).tolist() == [[0, 0, 0, 1, 1, 1, 2]]


def test_position_ids_restart_after_eos():
    x = _x([1, 2, EOS, 3, 4, EOS, 5], [EOS, 1, 2, 3, 4, 5, 6])
    assert position_ids_from_tokens(x, EOS).tolist() == [[0, 1, 2, 0, 1, 2, 0], [0, 0, 1, 2, 3, 4, 5]]


def test_no_eos_means_one_document():
    x = _x([1, 2, 3, 4])
    assert doc_ids_from_tokens(x, EOS).tolist() == [[0, 0, 0, 0]]
    assert position_ids_from_tokens(x, EOS).tolist() == [[0, 1, 2, 3]]


def test_split_window_shares_boundary_tokens():
    w = torch.arange(9)  # max_len 8, tokens 0..8
    out = split_window(w, 4)
    assert out.tolist() == [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]]


def test_split_window_batched():
    w = torch.arange(2 * 9).view(2, 9)
    out = split_window(w, 4)
    assert out.shape == (4, 5)
    assert out[0].tolist() == [0, 1, 2, 3, 4]
    assert out[2].tolist() == [9, 10, 11, 12, 13]


def test_split_window_requires_divisor():
    with pytest.raises(ValueError, match="must divide"):
        split_window(torch.arange(9), 3)
