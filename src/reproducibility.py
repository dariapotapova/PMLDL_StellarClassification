"""Project-wide deterministic execution helpers.

References
----------
PyTorch reproducibility notes:
https://docs.pytorch.org/docs/stable/notes/randomness.html

The implementation in this file is original project code.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

PROJECT_SEED = 42


def configure_determinism(seed: int = PROJECT_SEED) -> torch.Generator:
    """Seed supported RNGs and return a seeded DataLoader generator."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs in a PyTorch DataLoader worker."""

    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
