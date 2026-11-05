"""Run on a CUDA machine: `pytest -m gpu`. Skipped automatically elsewhere (see tests/conftest.py)."""
import dataclasses

import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.model.factory import build_model
from snapdragon.nn.functional.ce import chunked_ce, reference_ce
from snapdragon.optim.muon import newton_schulz

pytestmark = pytest.mark.gpu

CFG = ModelConfig(vocab_size=512, d_model=128, n_layers=3, n_heads=4, n_kv_heads=2, mlp_hidden=256, max_seq_len=64)


def test_forward_on_cuda_matches_cpu_in_fp32():
    torch.manual_seed(0)
    cpu = build_model(CFG).eval()
    gpu = build_model(CFG).to("cuda").eval()
    gpu.load_state_dict(cpu.state_dict())
    x = torch.randint(0, 512, (2, 32))
    with torch.no_grad():
        assert torch.allclose(cpu(x), gpu(x.cuda()).cpu(), atol=2e-3)


def test_bf16_autocast_forward_stays_close_to_fp32():
    m = build_model(CFG).to("cuda").eval()
    x = torch.randint(0, 512, (2, 32), device="cuda")
    with torch.no_grad():
        ref = m(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            low = m(x)
    assert (ref - low.float()).abs().max() < 0.25


def test_chunked_ce_gradients_on_cuda_match_the_reference():
    g = torch.Generator(device="cuda").manual_seed(0)
    h = torch.randn(300, 64, device="cuda", requires_grad=True, generator=g)
    w = torch.randn(1000, 64, device="cuda", requires_grad=True, generator=g)
    t = torch.randint(0, 1000, (300,), device="cuda", generator=g)
    ref = reference_ce(h, w, t, 1e-4)
    ref.backward()
    gh, gw = h.grad.clone(), w.grad.clone()
    h.grad = w.grad = None
    out = chunked_ce(h, w, t, 1e-4, chunk_tokens=64)
    out.backward()
    assert torch.allclose(out, ref, atol=1e-4)
    assert torch.allclose(h.grad, gh, atol=1e-4)
    assert torch.allclose(w.grad, gw, atol=1e-3)


def test_newton_schulz_on_cuda_flattens_the_spectrum():
    g = torch.randn(256, 512, device="cuda")
    s = torch.linalg.svdvals(newton_schulz(g).float())
    assert s.min() > 0.4
    assert s.max() < 1.6


def test_kv_cache_generation_on_cuda_equals_uncached():
    from snapdragon.inference.generate import stream_generate
    from snapdragon.inference.sampling import SamplingConfig

    m = build_model(CFG).to("cuda").eval()
    prompt = [3, 14, 15, 9]
    fast = list(stream_generate(m, prompt, 10, SamplingConfig(temperature=0)))
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(10):
            ids.append(int(m(torch.tensor([ids], device="cuda"))[0, -1].argmax()))
    assert fast == ids[len(prompt):]


@pytest.mark.parametrize("mode", ["none", "sac", "full"])
def test_activation_checkpointing_modes_agree_on_cuda(mode):
    torch.manual_seed(0)
    cfg = dataclasses.replace(CFG, act_ckpt=mode)
    m = build_model(cfg).to("cuda").train()
    x = torch.randint(0, 512, (2, 32), device="cuda")
    loss, _, _ = m(x, x)
    loss.backward()
    ref = build_model(CFG).to("cuda").train()
    ref.load_state_dict(m.state_dict())
    rloss, _, _ = ref(x, x)
    rloss.backward()
    assert torch.allclose(loss, rloss, atol=1e-4)
    for (n, p), (_, q) in zip(m.named_parameters(), ref.named_parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-3), n
