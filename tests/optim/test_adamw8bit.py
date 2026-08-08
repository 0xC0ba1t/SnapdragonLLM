import pytest
import torch

from snapdragon.optim.adamw8bit import AdamW8bit


def _problem(n=20_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(n, generator=g)
    scale = torch.exp(torch.randn(n, generator=g))  # ill-conditioned quadratic
    w = torch.nn.Parameter(torch.zeros(n))
    return w, target, scale


def _run(opt_cls, steps=300, **kw):
    w, target, scale = _problem()
    opt = opt_cls([w], lr=0.02, weight_decay=0.0, **kw)
    for _ in range(steps):
        opt.zero_grad()
        (0.5 * scale * (w - target) ** 2).mean().backward()
        opt.step()
    return w.detach(), (0.5 * scale * (w - target) ** 2).mean().item(), target


def test_tracks_fp32_adamw():
    w8, loss8, target = _run(AdamW8bit, betas=(0.9, 0.95))
    w32, loss32, _ = _run(torch.optim.AdamW, betas=(0.9, 0.95))
    start = (0.5 * torch.exp(torch.randn(20_000, generator=torch.Generator().manual_seed(0))) * target**2).mean().item()
    assert loss8 < 0.05 * start
    assert loss8 < 3 * loss32 + 1e-3
    corr = torch.corrcoef(torch.stack([w8, w32]))[0, 1]
    assert corr > 0.995


def test_small_tensors_keep_exact_fp32_state():
    w = torch.nn.Parameter(torch.zeros(100))
    opt = AdamW8bit([w])
    w.grad = torch.ones(100)
    opt.step()
    assert "exp_avg" in opt.state[w]
    assert "m_codes" not in opt.state[w]


def test_large_tensors_use_one_byte_state():
    w = torch.nn.Parameter(torch.zeros(1 << 16))
    opt = AdamW8bit([w])
    w.grad = torch.randn(1 << 16)
    opt.step()
    st = opt.state[w]
    nbytes = sum(t.numel() * t.element_size() for k, t in st.items() if torch.is_tensor(t))
    assert nbytes < 2.1 * w.numel()  # ~2 bytes/param vs 8 for fp32 Adam
    assert st["m_codes"].dtype == st["v_codes"].dtype == torch.uint8


def test_weight_decay_is_decoupled():
    w = torch.nn.Parameter(torch.ones(5000))
    w.grad = torch.zeros(5000)
    AdamW8bit([w], lr=0.1, weight_decay=0.5).step()
    assert torch.allclose(w, torch.full((5000,), 1 - 0.1 * 0.5), atol=1e-6)


def test_zero_gradient_keeps_state_finite():
    w = torch.nn.Parameter(torch.randn(8192))
    opt = AdamW8bit([w])
    for _ in range(3):
        w.grad = torch.zeros(8192)
        opt.step()
    assert torch.isfinite(w).all()


def test_bad_lr():
    with pytest.raises(ValueError, match="lr"):
        AdamW8bit([torch.nn.Parameter(torch.zeros(3))], lr=0)


def test_state_dict_roundtrip_continues_identically(tmp_path):
    def build():
        w, target, scale = _problem(n=10_000)
        return w, target, scale, AdamW8bit([w], lr=0.02, weight_decay=0.1)

    def advance(w, target, scale, opt, n):
        for _ in range(n):
            opt.zero_grad()
            (0.5 * scale * (w - target) ** 2).mean().backward()
            opt.step()

    w1, t, s, o1 = build()
    advance(w1, t, s, o1, 20)
    torch.save({"w": w1.detach(), "opt": o1.state_dict()}, tmp_path / "ck.pt")

    w2, _, _, o2 = build()
    ck = torch.load(tmp_path / "ck.pt")
    w2.data.copy_(ck["w"])
    o2.load_state_dict(ck["opt"])
    assert o2.state[w2]["m_codes"].dtype == torch.uint8

    advance(w1, t, s, o1, 10)
    advance(w2, t, s, o2, 10)
    assert torch.equal(w1, w2)  # bitwise identical, quantized state included


def test_offloaded_state_matches_resident_state():
    def run(**kw):
        w = torch.nn.Parameter(torch.zeros(8192))
        opt = AdamW8bit([w], lr=0.02, **kw)
        g = torch.Generator().manual_seed(0)
        for _ in range(5):
            w.grad = torch.randn(8192, generator=g)
            opt.step()
        return w.detach(), opt, w

    a, _, _ = run()
    b, opt, wb = run(state_device="cpu")
    assert torch.equal(a, b)
    assert opt.state[wb]["m_codes"].device.type == "cpu"
