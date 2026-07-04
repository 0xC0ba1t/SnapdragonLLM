import pytest
import torch

from snapdragon.nn.functional.ce import IGNORE_INDEX, chunked_ce_checkpointed, reference_ce


def _data(n=50, d=16, v=40, dtype=torch.float64, ignore=0):
    g = torch.Generator().manual_seed(0)
    h = torch.randn(n, d, generator=g, dtype=dtype, requires_grad=True)
    w = torch.randn(v, d, generator=g, dtype=dtype, requires_grad=True)
    t = torch.randint(0, v, (n,), generator=g)
    t[:ignore] = IGNORE_INDEX
    return h, w, t


def test_reference_matches_f_cross_entropy_without_z_loss():
    h, w, t = _data()
    ref = torch.nn.functional.cross_entropy((h @ w.T), t)
    assert torch.allclose(reference_ce(h, w, t), ref)


def test_z_loss_adds_positive_term():
    h, w, t = _data()
    assert reference_ce(h, w, t, 1e-2) > reference_ce(h, w, t, 0.0)


@pytest.mark.parametrize("z", [0.0, 1e-4])
@pytest.mark.parametrize("ignore", [0, 7])
def test_chunked_matches_reference_forward_and_backward(z, ignore):
    h, w, t = _data(ignore=ignore)
    ref = reference_ce(h, w, t, z)
    ref.backward()
    gh, gw = h.grad.clone(), w.grad.clone()
    h.grad = w.grad = None
    out = chunked_ce_checkpointed(h, w, t, z, chunk_tokens=16)
    out.backward()
    assert torch.allclose(out, ref, atol=1e-10)
    assert torch.allclose(h.grad, gh, atol=1e-10)
    assert torch.allclose(w.grad, gw, atol=1e-10)


def test_all_ignored_is_zero_not_nan():
    h, w, t = _data(n=8, ignore=8)
    out = chunked_ce_checkpointed(h, w, t, 1e-4, chunk_tokens=4)
    assert out.item() == 0.0


from snapdragon.nn.functional.ce import chunked_ce  # noqa: E402


@pytest.mark.parametrize("z", [0.0, 1e-4, 1e-2])
@pytest.mark.parametrize("chunk", [7, 16, 1000])
@pytest.mark.parametrize("ignore", [0, 9])
def test_custom_function_matches_reference(z, chunk, ignore):
    h, w, t = _data(ignore=ignore)
    ref = reference_ce(h, w, t, z)
    ref.backward()
    gh, gw = h.grad.clone(), w.grad.clone()
    h.grad = w.grad = None
    out = chunked_ce(h, w, t, z, chunk_tokens=chunk)
    out.backward()
    assert torch.allclose(out, ref, atol=1e-10)
    assert torch.allclose(h.grad, gh, atol=1e-10)
    assert torch.allclose(w.grad, gw, atol=1e-10)


def test_gradcheck():
    g = torch.Generator().manual_seed(1)
    h = torch.randn(6, 4, generator=g, dtype=torch.float64, requires_grad=True)
    w = torch.randn(9, 4, generator=g, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([1, 2, IGNORE_INDEX, 4, 0, 8])
    assert torch.autograd.gradcheck(lambda a, b: chunked_ce(a, b, t, 1e-2, 4), (h, w))


def test_bf16_hidden_with_fp32_weight_runs_and_returns_matching_grad_dtypes():
    g = torch.Generator().manual_seed(2)
    h = torch.randn(32, 16, generator=g).bfloat16().requires_grad_()
    w = torch.randn(40, 16, generator=g, requires_grad=True)
    t = torch.randint(0, 40, (32,), generator=g)
    loss = chunked_ce(h, w, t, 1e-4, chunk_tokens=8)
    loss.backward()
    assert h.grad.dtype == torch.bfloat16
    assert w.grad.dtype == torch.float32
    ref = reference_ce(h.float(), w.to(torch.bfloat16).float(), t, 1e-4)
    assert abs(loss.item() - ref.item()) < 5e-2


def test_only_the_lse_vector_is_saved_not_the_logits():
    h, w, t = _data(n=64, v=200)
    saved = []

    def pack(x):
        saved.append(x.numel())
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        chunked_ce(h, w, t, 1e-4, chunk_tokens=16)
    assert max(saved) < 64 * 200  # nothing as large as the (N, V) logits is retained
