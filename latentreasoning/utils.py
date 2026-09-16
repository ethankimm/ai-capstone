"""Small shared helpers."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> int:
    """Seed python/numpy/torch (torch only if installed). Record the value in
    `RunRecord.seed`."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
