#!/usr/bin/env python3
# encoding: utf-8

"""Wrapper de compatibilidade para a geracao de backbone via pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.backbone.generation import (
    SUPPORTED_METHODS,
    build_parser,
    main,
    run_backbone_generation,
)

__all__ = [
    "SUPPORTED_METHODS",
    "build_parser",
    "main",
    "run_backbone_generation",
]


if __name__ == "__main__":
    main()
