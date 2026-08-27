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
