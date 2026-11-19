"""Mixed precision is the production path (bf16 autocast, fp32 master weights); the other trainer tests use
fp32 for exactness, so this file makes sure the bf16 path actually trains."""
import pytest
import torch


@pytest.mark.parametrize("act_ckpt", ["none", "sac", "full"])
def test_bf16_training_learns_with_every_checkpointing_mode(make_trainer, tiny_config, act_ckpt):
    cfg = tiny_config(precision="bf16", z_loss_coeff=1e-4)
    cfg.model.act_ckpt = act_ckpt
    t = make_trainer(cfg)
    first = t.train_step()["loss"]
    last = first
    for _ in range(25):
        last = t.train_step()["loss"]
    assert torch.isfinite(torch.tensor(last))
    assert last < 0.85 * first
    assert t.state.skipped_steps == 0


def test_master_weights_and_grads_stay_fp32_under_bf16_autocast(make_trainer, tiny_config):
    t = make_trainer(tiny_config(precision="bf16"))
    t.train_step()
    assert {p.dtype for p in t.raw_model.parameters()} == {torch.float32}
    t.optimizer.zero_grad(set_to_none=False)
    batch = next(t.train_loader)
    from snapdragon.train.precision import autocast_context

    with autocast_context("bf16", torch.device("cpu")):
        loss, _, _ = t.model(batch.x, batch.y)
    loss.backward()
    assert {p.grad.dtype for p in t.raw_model.parameters()} == {torch.float32}


def test_bf16_and_fp32_agree_closely_on_the_first_loss(make_trainer, tiny_config):
    f32 = make_trainer(tiny_config(precision="fp32")).train_step()["loss"]
    b16 = make_trainer(tiny_config(precision="bf16")).train_step()["loss"]
    assert b16 == pytest.approx(f32, abs=0.05)
