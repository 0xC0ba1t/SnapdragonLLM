from snapdragon.train.callbacks.base import Callback, CallbackList


class Recorder(Callback):
    def __init__(self, name, log, priority=50):
        self.name, self.log, self.priority = name, log, priority

    def on_train_start(self, trainer):
        self.log.append((self.name, "start"))

    def on_step_end(self, trainer, metrics):
        self.log.append((self.name, "step"))
        metrics[self.name] = True

    def on_train_end(self, trainer):
        self.log.append((self.name, "end"))


def test_dispatch_order_follows_priority_then_insertion():
    log = []
    cbs = CallbackList([Recorder("logger", log, 50), Recorder("producer", log, 10), Recorder("ckpt", log, 90),
                        Recorder("logger2", log, 50)])
    cbs.on_step_end(None, {})
    assert [n for n, _ in log] == ["producer", "logger", "logger2", "ckpt"]


def test_metrics_dict_is_shared_so_loggers_see_producer_values():
    metrics = {}
    CallbackList([Recorder("a", [], 10), Recorder("b", [], 20)]).on_step_end(None, metrics)
    assert metrics == {"a": True, "b": True}


def test_lifecycle_and_add():
    log = []
    cbs = CallbackList()
    cbs.add(Recorder("late", log, 80))
    cbs.add(Recorder("early", log, 5))
    cbs.on_train_start(None)
    cbs.on_train_end(None)
    assert log == [("early", "start"), ("late", "start"), ("early", "end"), ("late", "end")]
    assert len(cbs) == 2


def test_base_callback_hooks_are_noops():
    cb = Callback()
    cb.on_train_start(None)
    cb.on_step_end(None, {})
    cb.on_train_end(None)
    cb.on_exception(None, RuntimeError())
