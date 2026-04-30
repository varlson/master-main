from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pipeline.bootstrap import WORKSPACE_ROOT


DEFAULT_ANALYSIS_OUTPUT_ROOT = WORKSPACE_ROOT / "outputs" / "analysis"
DEFAULT_BACKBONE_ANALYSIS_ROOT = WORKSPACE_ROOT / "outputs" / "backbone"


def run_backbone_analysis(
    *,
    datasets: list[str] | None = None,
    requested_names: list[str] | None = None,
    methods: list[str] | None = None,
    alpha: float | None = None,
    original_path: Path | None = None,
    backbone_paths: list[Path] | None = None,
    dataset_label: str = "custom",
    top_k: int = 20,
    robustness_steps: int = 20,
    random_trials: int = 10,
    output_root: Path | None = None,
    output_tag: str | None = None,
):
    from pipeline.backbone.analysis_runtime import run_analysis_pipeline

    resolved_output_root = output_root
    if resolved_output_root is None and output_tag:
        resolved_output_root = DEFAULT_ANALYSIS_OUTPUT_ROOT / output_tag
    elif resolved_output_root is None:
        resolved_output_root = DEFAULT_ANALYSIS_OUTPUT_ROOT / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

    return run_analysis_pipeline(
        datasets=datasets,
        requested_names=requested_names,
        methods=methods,
        alpha=alpha,
        original_path=original_path,
        backbone_paths=backbone_paths,
        dataset_label=dataset_label,
        top_k=top_k,
        robustness_steps=robustness_steps,
        random_trials=random_trials,
        output_root=resolved_output_root,
        output_tag=output_tag,
    )


def run_single_backbone_analysis(
    *,
    dataset: str,
    original_path: Path,
    backbone_path: Path,
    output_root: Path | None = None,
    top_k: int = 20,
    robustness_steps: int = 20,
    random_trials: int = 10,
):
    from pipeline.backbone.analysis_runtime import (
        run_single_backbone_analysis as runtime_single_backbone_analysis,
    )

    return runtime_single_backbone_analysis(
        dataset=dataset,
        original_path=original_path,
        backbone_path=backbone_path,
        output_root=output_root,
        top_k=top_k,
        robustness_steps=robustness_steps,
        random_trials=random_trials,
    )
