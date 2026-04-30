from __future__ import annotations

from pathlib import Path

from pipeline.backbone.analysis import (
    DEFAULT_ANALYSIS_OUTPUT_ROOT,
    run_backbone_analysis,
)


def run_backbone_analysis_workflow(
    *,
    datasets: list[str] | None,
    network_names: list[str] | None,
    methods: list[str],
    alpha: float | None,
    original_path: str | None,
    backbone_paths: list[str] | None,
    dataset_label: str,
    top_k: int,
    robustness_steps: int,
    random_trials: int,
    output_tag: str | None,
) -> None:
    run_backbone_analysis(
        datasets=datasets,
        requested_names=network_names,
        methods=methods,
        alpha=alpha,
        original_path=Path(original_path) if original_path else None,
        backbone_paths=[Path(path) for path in backbone_paths]
        if backbone_paths
        else None,
        dataset_label=dataset_label,
        top_k=top_k,
        robustness_steps=robustness_steps,
        random_trials=random_trials,
        output_tag=output_tag,
    )
    print(f"Artefatos de analise em: {DEFAULT_ANALYSIS_OUTPUT_ROOT}")
