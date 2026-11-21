import dataclasses

import pytest

from snapdragon.config.model import ModelConfig
from snapdragon.config.schema import Config
from snapdragon.train.memory import (
    activation_bytes_per_token_per_layer,
    estimate_memory,
)


def cfg_500m(act_ckpt="sac", mb=4, offload=False):
    c = Config()
    c.model = dataclasses.replace(ModelConfig(), act_ckpt=act_ckpt)
    c.train.micro_batch_size = mb
    c.optim.offload_states = offload
    return c


def test_total_is_the_sum_of_parts():
    e = estimate_memory(cfg_500m())
    assert e.total == pytest.approx(sum(v for k, v in e.rows() if k != "TOTAL"))


def test_500m_with_sac_fits_a_16gb_card():
    assert estimate_memory(cfg_500m("sac", mb=4)).total < 14.0


def test_500m_with_full_checkpointing_fits_12gb():
    assert estimate_memory(cfg_500m("full", mb=2)).total < 11.5


def test_weights_and_grads_are_4_bytes_per_param():
    e = estimate_memory(cfg_500m())
    n = ModelConfig().num_params()
    assert e.weights == pytest.approx(n * 4 / 1e9, rel=1e-6)
    assert e.grads == e.weights


def test_checkpointing_trades_activation_memory_in_order():
    none = estimate_memory(cfg_500m("none")).activations
    sac = estimate_memory(cfg_500m("sac")).activations
    full = estimate_memory(cfg_500m("full")).activations
    assert full < sac < none


def test_selective_sits_between_full_and_none():
    c = cfg_500m("selective")
    c.model.act_ckpt_every = 2
    mid = estimate_memory(c).activations
    assert estimate_memory(cfg_500m("full")).activations < mid < estimate_memory(cfg_500m("none")).activations


def test_offload_removes_optimizer_state_from_the_gpu_budget():
    assert estimate_memory(cfg_500m(offload=True)).optimizer == 0.0
    assert estimate_memory(cfg_500m(offload=False)).optimizer > 1.0


def test_activations_scale_with_micro_batch():
    a1 = estimate_memory(cfg_500m(mb=1)).activations
    a4 = estimate_memory(cfg_500m(mb=4)).activations
    assert a4 == pytest.approx(4 * a1, rel=1e-6)


def test_per_layer_activation_ordering():
    m = ModelConfig()
    none = activation_bytes_per_token_per_layer(m, "none")
    assert activation_bytes_per_token_per_layer(m, "full") < activation_bytes_per_token_per_layer(m, "sac") < none


def test_default_ce_chunk_keeps_logit_memory_under_a_gigabyte():
    # 2048 tokens x 32768 vocab x (fp32 logits + fp32 grad + bf16 copy)
    assert estimate_memory(cfg_500m()).logits < 1.0


def test_weight_ema_adds_one_fp32_copy_to_the_budget():
    base = estimate_memory(cfg_500m())
    c = cfg_500m()
    c.train.ema_decay = 0.999
    with_ema = estimate_memory(c)
    assert with_ema.ema == pytest.approx(base.weights)
    assert with_ema.total == pytest.approx(base.total + base.weights)
    assert any(name.startswith("weight EMA") for name, _ in with_ema.rows())
    assert not any(name.startswith("weight EMA") for name, _ in base.rows())


def test_fp32_precision_roughly_doubles_activations_and_no_longer_fits_16gb():
    """Measured on a Colab T4: the 16 GB preset in fp32 ran out of memory (14.5 GB usable), bf16 peaked at
    13.5 GB. The estimator must reflect that fp32 is much heavier than bf16."""
    bf16 = estimate_memory(cfg_500m())
    c = cfg_500m()
    c.train.precision = "fp32"
    fp32 = estimate_memory(c)
    assert fp32.activations == pytest.approx(2 * bf16.activations)
    assert fp32.logits > bf16.logits
    assert fp32.weights == bf16.weights  # master weights are fp32 either way
    assert bf16.total < 14.5 < fp32.total
