from __future__ import annotations


def alpha_cut_name(alpha: float) -> str:
    return f"alpah_filter{str(alpha).replace('.', '_')}"


def backbone_dataset_name(dataset_name: str, method: str, alpha: float) -> str:
    return f"{dataset_name}-by-{method}-with-{alpha_cut_name(alpha)}"
