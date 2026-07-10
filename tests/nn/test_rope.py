import torch

from snapdragon.nn.rope import RotaryEmbedding, apply_rotary, rope_tables


def test_table_shapes_and_first_position_is_identity():
    cos, sin = rope_tables(8, 16, 10_000.0)
    assert cos.shape == sin.shape == (16, 4)
    assert torch.allclose(cos[0], torch.ones(4))
    assert torch.allclose(sin[0], torch.zeros(4))


def test_rotation_preserves_norm():
    rope = RotaryEmbedding(16, 32)
    q, k = torch.randn(2, 4, 10, 16), torch.randn(2, 2, 10, 16)
    rq, rk = rope(q, k)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    assert torch.allclose(rk.norm(dim=-1), k.norm(dim=-1), atol=1e-5)


def test_attention_scores_depend_only_on_relative_position():
    d = 32
    rope = RotaryEmbedding(d, 64)
    q = torch.randn(1, 1, 1, d)
    k = torch.randn(1, 1, 1, d)

    def score(m, n):
        qm = apply_rotary(q, rope.cos[m : m + 1], rope.sin[m : m + 1])
        kn = apply_rotary(k, rope.cos[n : n + 1], rope.sin[n : n + 1])
        return (qm * kn).sum()

    assert torch.allclose(score(5, 2), score(20, 17), atol=1e-4)
    assert torch.allclose(score(3, 3), score(40, 40), atol=1e-4)
    assert not torch.allclose(score(5, 2), score(5, 3), atol=1e-4)


def test_default_positions_equal_explicit_arange():
    rope = RotaryEmbedding(16, 32)
    q, k = torch.randn(2, 4, 10, 16), torch.randn(2, 2, 10, 16)
    pos = torch.arange(10).expand(2, 10)
    a = rope(q, k)
    b = rope(q, k, pos)
    assert torch.allclose(a[0], b[0], atol=1e-6)
    assert torch.allclose(a[1], b[1], atol=1e-6)


def test_position_ids_restart_per_document():
    rope = RotaryEmbedding(16, 32)
    q = torch.randn(1, 1, 4, 16)
    k = q.clone()
    pos = torch.tensor([[0, 1, 0, 1]])
    rq, _ = rope(q, k, pos)
    rq0, _ = rope(q[..., :2, :].contiguous(), k[..., :2, :].contiguous())
    # token 2 sits at position 0 again, so it is rotated like token 0 would be
    assert torch.allclose(rq[0, 0, 2], apply_rotary(q[0, 0, 2], rope.cos[0], rope.sin[0]), atol=1e-6)
    assert rq0.shape[-2] == 2


def test_buffers_are_not_in_state_dict():
    assert list(RotaryEmbedding(8, 8).state_dict()) == []


def test_bf16_in_bf16_out():
    rope = RotaryEmbedding(16, 32)
    q = torch.randn(1, 2, 8, 16).bfloat16()
    rq, rk = rope(q, q)
    assert rq.dtype == rk.dtype == torch.bfloat16


def test_rebuild_restores_tables_after_to_empty():
    ref = RotaryEmbedding(16, 32)
    with torch.device("meta"):
        meta = RotaryEmbedding(16, 32)
    assert meta.cos.device.type == "meta"
    meta.to_empty(device="cpu")
    meta.rebuild("cpu")
    assert torch.equal(meta.cos, ref.cos)
    assert torch.equal(meta.sin, ref.sin)
