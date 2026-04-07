from __future__ import annotations

import random

import numpy as np
import torch


def parse_seeds(raw_value) -> list[int]:
    if raw_value is None:
        return [42]

    if isinstance(raw_value, int):
        return [int(raw_value)]

    if isinstance(raw_value, str):
        seeds = [item.strip() for item in raw_value.split(",") if item.strip()]
        return [int(seed) for seed in seeds] if seeds else [42]

    if isinstance(raw_value, list):
        parsed = [int(item) for item in raw_value]
        return parsed if parsed else [42]

    raise ValueError(f"Lista de seeds invalida: {raw_value!r}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
