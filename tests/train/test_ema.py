import pytest
import torch

from snapdragon.checkpoint.io import load_checkpoint
from snapdragon.train.checkpointing import restore_trainer, save_trainer_checkpoint
from snapdragon.train.ema import EMACallback, WeightEMA


def _model():
    torch.manual_seed(0)
    return torch.nn.Linear(3, 2)


def test_update_matches_the_closed_form_without_warmup():
    m = _model()
    ema = WeightEMA(m, decay=0.9, warmup=False)
    start = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update(m)
    assert torch.allclose(ema.shadow["weight"], 0.9 * start + 0.1 * m.weight, atol=1e-6)


def test_warmup_ramps_the_decay():
    ema = WeightEMA(_model(), decay=0.999)
    assert ema.current_decay() == pytest.approx(0.1)  # (1+0)/(10+0)
    ema.num_updates = 90
    assert ema.current_decay() == pytest.approx(0.91)
    ema.num_updates = 10**6
    assert ema.current_decay() == 0.999


def test_applied_swaps_in_and_restores_the_live_weights():
    m = _model()
    ema = WeightEMA(m, decay=0.5, warmup=False)
    live = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.fill_(7.0)
    ema.update(m)
    shadow = ema.shadow["weight"].clone()
    with ema.applied(m):
        assert torch.allclose(m.weight, shadow)
    assert torch.equal(m.weight, torch.full_like(live, 7.0))


def test_applied_restores_even_if_the_body_raises():
    m = _model()
    ema = WeightEMA(m)
    with torch.no_grad():
        m.weight.fill_(2.0)
    with pytest.raises(RuntimeError), ema.applied(m):
        raise RuntimeError("boom")
    assert torch.equal(m.weight, torch.full_like(m.weight, 2.0))


def test_shadow_stays_fp32_even_for_bf16_weights():
    m = _model().bfloat16()
    ema = WeightEMA(m)
    assert ema.shadow["weight"].dtype == torch.float32
    ema.update(m)
    assert ema.shadow["weight"].dtype == torch.float32


def test_invalid_decay():
    with pytest.raises(ValueError, match="decay"):
        WeightEMA(_model(), decay=1.0)


def test_callback_tracks_training_skips_bad_steps_and_is_checkpointed(make_trainer, tiny_config, tmp_path):
    cfg = tiny_config()
    cfg.train.out_dir = str(tmp_path)
    cb = EMACallback(decay=0.9)
    t = make_trainer(cfg, callbacks=[cb])
    t.callbacks.on_train_start(t)
    for step in range(1, 6):
        t.callbacks.on_step_end(t, {**t.train_step(), "step": step})
    assert cb.ema.num_updates == 5
    before = cb.ema.num_updates
    cb.on_step_end(t, {"step": 6, "skipped": True})
    assert cb.ema.num_updates == before  # skipped steps do not move the average

    path = save_trainer_checkpoint(t)
    assert "extra_ema" in load_checkpoint(path)
    cb2 = EMACallback(decay=0.9)
    t2 = make_trainer(cfg, callbacks=[cb2])
    t2.callbacks.on_train_start(t2)
    restore_trainer(t2, load_checkpoint(path))
    assert cb2.ema.num_updates == 5
    assert torch.equal(cb2.ema.shadow["transformer.embed.weight"], cb.ema.shadow["transformer.embed.weight"])
