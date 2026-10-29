"""One place to seed every RNG the training loop touches.

``world.seed`` used to reach ``numpy``/``random`` only (and only on the
CARLA path, inside the vendored scenario manager) — PyTorch was never
seeded, so network initialisation and stochastic-policy sampling differed
between two runs of the same config and seed. A recorded seed then pinned
the traffic replay but not the learner, which defeats multi-seed reporting.

Call :func:`seed_everything` once, before any algorithm is constructed.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
from loguru import logger


def seed_everything(seed: Optional[int], *, deterministic_torch: bool = False) -> Optional[int]:
    """Seed ``random``, ``numpy``, and ``torch`` (CPU + all CUDA devices).

    ``deterministic_torch=True`` additionally asks cuDNN for deterministic
    kernels (slower; opt-in via ``runtime.deterministic_torch``). Returns
    the seed actually applied, or None when seeding is disabled.
    """
    if seed is None:
        return None
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Reaches child processes only: this interpreter's string hashing is fixed at
    # start-up, so code here must never depend on ``hash(str)`` or set order.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        pass
    logger.info(f"Seeded random/numpy/torch with seed={seed}"
                + (" (deterministic cuDNN)" if deterministic_torch else ""))
    return seed


__all__ = ["seed_everything"]
