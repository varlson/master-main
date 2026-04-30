from __future__ import annotations

import sys
from pathlib import Path


REFAT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = REFAT_DIR.parent
PRIMARY_DATA_ROOT = WORKSPACE_ROOT / "data" / "npy"
LEGACY_PROJECT_ROOT = WORKSPACE_ROOT / "legacy" / "old-proj"


def ensure_workspace_root_on_path() -> Path:
    workspace_path = str(WORKSPACE_ROOT)
    if workspace_path not in sys.path:
        sys.path.insert(0, workspace_path)
    return WORKSPACE_ROOT


def ensure_legacy_project_on_path() -> Path:
    # Alias de compatibilidade: o pipeline agora prioriza os modulos ativos em
    # WORKSPACE_ROOT, enquanto old-proj fica apenas como area legada.
    return ensure_workspace_root_on_path()
