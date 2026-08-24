import pytest
import torch

from snapdragon.train.callbacks.base import Callback


def test_loss_goes_down(make_trainer):
    t = make_trainer()
    first = t.train_step()["loss"]
    last = first
    for _ in range(25):
        last = t.train_step()["loss"]
    assert last < 0.8 * first


def test_step_and_token_accounting(make_trainer):
    t = make_trainer()
    m = t.train_step()
    assert m["step"] == 1
    assert t.state.tokens_seen == t.cfg.train.global_batch_tokens
    assert m["tokens_seen"] == t.cfg.train.global_batch_tokens
    assert m["skipped"] is False


def test_lr_follows_schedule(make_trainer):
    t = make_trainer()
    lrs = [t.train_step()["lr"] for _ in range(4)]
    base = t.optimizer.param_groups[0]["base_lr"]
    assert lrs[0] == pytest.approx(base * 0.5)  # warmup is 2 steps: step 0 runs at 1/2 of base
    assert lrs[1] == pytest.approx(base)


def test_fit_runs_to_total_steps_and_fires_callbacks(make_trainer):
    events = []

    class Spy(Callback):
        def on_train_start(self, trainer):
            events.append("start")

        def on_step_end(self, trainer, metrics):
            events.append(metrics["step"])

        def on_train_end(self, trainer):
            events.append("end")

    t = make_trainer(callbacks=[Spy()])
    state = t.fit()
    assert state.step == t.total_steps == 30
    assert events[0] == "start"
    assert events[-1] == "end"
    assert events[1:-1] == list(range(1, 31))


def test_stop_requested_ends_the_loop_early(make_trainer):
    class Stopper(Callback):
        def on_step_end(self, trainer, metrics):
            if metrics["step"] == 3:
                trainer.stop_requested = True

    t = make_trainer(callbacks=[Stopper()])
    assert t.fit().step == 3


def test_exceptions_reach_on_exception_hook_and_propagate(make_trainer):
    seen = []

    class Boom(Callback):
        def on_step_end(self, trainer, metrics):
            raise RuntimeError("kaboom")

        def on_exception(self, trainer, exc):
            seen.append(str(exc))

    t = make_trainer(callbacks=[Boom()])
    try:
        t.fit()
    except RuntimeError:
        pass
    else:
        raise AssertionError("exception should propagate")
    assert seen == ["kaboom"]


def test_non_finite_gradients_skip_the_step_and_keep_weights(make_trainer):
    t = make_trainer()
    before = {n: p.detach().clone() for n, p in t.raw_model.named_parameters()}
    original_forward = t.raw_model.forward

    def poisoned(*args, **kwargs):
        loss, ce, z = original_forward(*args, **kwargs)
        return loss * float("nan"), ce, z

    t.raw_model.forward = poisoned
    m = t.train_step()
    assert m["skipped"] is True
    assert t.state.skipped_steps == 1
    for n, p in t.raw_model.named_parameters():
        assert torch.equal(p, before[n]), n
    assert t.state.step == 1  # the step counter (and schedule) still advance


def test_loss_spike_is_skipped(make_trainer, tiny_config):
    cfg = tiny_config(loss_spike_factor=1.5)
    t = make_trainer(cfg)
    for _ in range(5):
        t.train_step()
    ema = t.state.loss_ema
    original_forward = t.raw_model.forward

    def spiky(*args, **kwargs):
        loss, ce, z = original_forward(*args, **kwargs)
        return loss * 100.0, ce, z

    t.raw_model.forward = spiky
    m = t.train_step()
    assert m["skipped"] is True
    assert t.state.loss_ema == ema  # the spike must not contaminate the EMA


def test_sequence_length_warmup_grows_while_tokens_per_step_stay_constant(make_trainer, tiny_config):
    cfg = tiny_config(seq_len_start=4, seq_len_warmup_tokens=64 * 10)  # 64 tokens/step, ramp over 10 steps
    t = make_trainer(cfg)
    lens, tokens = [], []
    for _ in range(14):
        before = t.state.tokens_seen
        lens.append(t.train_step()["seq_len"])
        tokens.append(t.state.tokens_seen - before)
    assert lens[0] == 4
    assert lens[-1] == 16
    assert lens == sorted(lens)
    assert set(tokens) == {64}
