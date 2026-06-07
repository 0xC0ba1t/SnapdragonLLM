import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
