import logging

from snapdragon.utils import logging as sd_logging


def test_rank_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    assert sd_logging.get_rank() == 0
    assert sd_logging.is_main_process()


def test_non_main_rank_is_quiet(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    sd_logging.setup_logging(logging.DEBUG, force=True)
    assert logging.getLogger().level == logging.WARNING
    assert not sd_logging.is_main_process()
    monkeypatch.setenv("RANK", "0")
    sd_logging.setup_logging(logging.INFO, force=True)
    assert logging.getLogger().level == logging.INFO
