import pytest
import torch
import torch.nn.functional as F

from snapdragon.nn.sdpa import resolve_backend, sdpa_backend


def test_default_is_auto(monkeypatch):
    monkeypatch.delenv("SNAPDRAGON_SDPA", raising=False)
    assert resolve_backend() == "auto"


def test_env_override(monkeypatch):
    monkeypatch.setenv("SNAPDRAGON_SDPA", "math")
    assert resolve_backend() == "math"
    assert resolve_backend("efficient") == "efficient"


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown sdpa backend"):
        resolve_backend("turbo")


@pytest.mark.parametrize("name", ["auto", "math"])
def test_context_runs_attention(name):
    q = torch.randn(1, 2, 8, 16)
    with sdpa_backend(name):
        out = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    assert out.shape == q.shape
