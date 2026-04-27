from __future__ import annotations

import sys
from pathlib import Path


REFAT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = REFAT_DIR.parent
LEGACY_PROJECT_ROOT = WORKSPACE_ROOT / "old-proj"


def ensure_legacy_project_on_path() -> Path:
    legacy_path = str(LEGACY_PROJECT_ROOT)
    if legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)
    return LEGACY_PROJECT_ROOT

