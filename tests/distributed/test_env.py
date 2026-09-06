import pytest
import torch

from snapdragon.distributed.env import DistEnv, detect_env, device_for, init_distributed


def test_defaults_describe_a_single_process():
    env = detect_env({})
    assert env == DistEnv(0, 0, 1)
    assert not env.distributed
    assert env.is_main


def test_reads_torchrun_variables():
    env = detect_env({"RANK": "5", "LOCAL_RANK": "1", "WORLD_SIZE": "8"})
    assert (env.rank, env.local_rank, env.world_size) == (5, 1, 8)
    assert env.distributed
    assert not env.is_main


def test_inconsistent_rank_is_rejected():
    with pytest.raises(ValueError, match="outside WORLD_SIZE"):
        detect_env({"RANK": "4", "WORLD_SIZE": "4"})


def test_single_process_init_is_a_noop():
    env = init_distributed(DistEnv())
    assert env == DistEnv()
    assert not torch.distributed.is_initialized()


def test_device_for_cpu_only_machine():
    if not torch.cuda.is_available():
        assert device_for(DistEnv(1, 1, 2)) == torch.device("cpu")
