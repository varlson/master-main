from __future__ import annotations

from pipeline.backbone.generation import DEFAULT_BACKBONE_OUTPUT_ROOT, run_backbone_generation


def run_backbone_generation_workflow(
    *,
    datasets: list[str] | None,
    methods: list[str] | None,
    alpha: float | None,
    min_degree: int | None,
) -> None:
    generated = run_backbone_generation(
        datasets=datasets,
        methods=methods,
        alpha=alpha,
        min_degree=min_degree,
    )
    print(f"Backbones gerados: {len(generated)}")
    print(f"Artefatos de backbone em: {DEFAULT_BACKBONE_OUTPUT_ROOT}")
