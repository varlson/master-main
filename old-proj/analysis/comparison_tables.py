from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd


DEFAULT_MODEL_ORDER = ["STICformer", "MTGNN", "GraphWaveNet"]
DATASET_LABELS = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY"}
BACKBONE_LABELS = {
    "original": "Full",
    "disp_fil": "DF",
    "nois_corr": "NC",
    "high_sal": "HSS",
}
VARIANT_ORDER = {"Full": 0, "DF": 1, "NC": 2, "HSS": 3}
METRIC_COLUMN_MAP = {
    "MAE": "test_mae_mean",
    "RMSE": "test_rmse_mean",
    "WAPE": "test_wape_mean",
}


def base_dataset_name(dataset: str) -> str:
    return str(dataset).split("-by-", 1)[0]


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, str(dataset).upper())


def backbone_method_name(dataset: str) -> str:
    dataset = str(dataset)
    match = re.search(r"-by-(.*?)-with-", dataset)
    if match:
        return match.group(1)
    if "-by-" in dataset:
        return dataset.split("-by-", 1)[1]
    return "original"


def graph_variant(dataset: str, family: str) -> str:
    if family == "original":
        return "Full"
    return BACKBONE_LABELS.get(backbone_method_name(dataset), backbone_method_name(dataset))


def load_family_results(csv_path: str | Path, family: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame["experiment_family"] = family
    frame["base_dataset"] = frame["dataset"].map(base_dataset_name)
    frame["dataset_label"] = frame["base_dataset"].map(dataset_label)
    frame["graph_variant"] = [graph_variant(dataset, family) for dataset in frame["dataset"]]
    frame["backbone"] = frame["graph_variant"]

    for metric_name, column_name in METRIC_COLUMN_MAP.items():
        frame[metric_name] = frame[column_name].astype(float)

    return frame


def sort_models(models: list[str]) -> list[str]:
    preferred = {model: index for index, model in enumerate(DEFAULT_MODEL_ORDER)}
    return sorted(models, key=lambda model: (preferred.get(model, len(preferred)), model))


def sort_variants(variants: list[str]) -> list[str]:
    return sorted(variants, key=lambda variant: (VARIANT_ORDER.get(variant, len(VARIANT_ORDER)), variant))


def _dataset_filename(base_dataset: str) -> str:
    return str(base_dataset).strip().lower()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower())
    return slug.strip("-")


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_metric(value: Any) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.3f}"


def _format_delta(value: Any, *, latex: bool) -> str:
    if pd.isna(value):
        return "--"
    number = float(value)
    sign = "+" if number > 0 else ""
    suffix = r"\%" if latex else "%"
    return f"{sign}{number:.2f}{suffix}"


def _metric_value(row: dict[str, Any] | None, metric_name: str) -> float:
    if row is None:
        return float("nan")
    value = row.get(metric_name)
    if pd.isna(value):
        return float("nan")
    return float(value)


def _delta_mae(original_mae: float, comparison_mae: float) -> float:
    if pd.isna(original_mae) or pd.isna(comparison_mae) or abs(float(original_mae)) < 1e-12:
        return float("nan")
    return ((float(comparison_mae) - float(original_mae)) / float(original_mae)) * 100.0


def _model_rows(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty:
        return {}
    rows = frame.to_dict("records")
    return {str(row["model"]): row for row in rows}


def _build_side_by_side_frame(
    original_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    comparison_label: str,
) -> pd.DataFrame:
    original_rows = _model_rows(original_frame)
    comparison_rows = _model_rows(comparison_frame)
    models = sort_models(sorted(set(original_rows) | set(comparison_rows)))

    rows: list[dict[str, Any]] = []
    for model in models:
        original_row = original_rows.get(model)
        comparison_row = comparison_rows.get(model)
        original_mae = _metric_value(original_row, "MAE")
        comparison_mae = _metric_value(comparison_row, "MAE")
        rows.append(
            {
                "Model": model,
                "Original MAE": original_mae,
                f"{comparison_label} MAE": comparison_mae,
                "Original RMSE": _metric_value(original_row, "RMSE"),
                f"{comparison_label} RMSE": _metric_value(comparison_row, "RMSE"),
                "Original WAPE": _metric_value(original_row, "WAPE"),
                f"{comparison_label} WAPE": _metric_value(comparison_row, "WAPE"),
                "Delta MAE (%)": _delta_mae(original_mae, comparison_mae),
            }
        )

    return pd.DataFrame(rows)


def _aggregate_dataset_frame(frame: pd.DataFrame, base_dataset: str) -> pd.DataFrame:
    dataset_frame = frame[frame["base_dataset"] == base_dataset].copy()
    if dataset_frame.empty:
        return dataset_frame

    return (
        dataset_frame.groupby(
            ["base_dataset", "dataset_label", "dataset", "backbone", "model"],
            as_index=False,
        )[["MAE", "RMSE", "WAPE"]]
        .mean()
    )


def _source_dataset_name(frame: pd.DataFrame, backbone: str) -> str | None:
    matches = (
        frame.loc[frame["backbone"] == backbone, "dataset"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if not matches:
        return None
    return sorted(matches)[0]


def _build_dataset_bundle(frame: pd.DataFrame, base_dataset: str) -> dict[str, Any] | None:
    dataset_frame = _aggregate_dataset_frame(frame, base_dataset)
    if dataset_frame.empty:
        return None

    original_frame = dataset_frame[dataset_frame["backbone"] == "Full"].copy()
    backbone_frame = dataset_frame[dataset_frame["backbone"] != "Full"].copy()
    if original_frame.empty or backbone_frame.empty:
        return None

    dataset_title = str(dataset_frame["dataset_label"].dropna().iloc[0])
    comparisons: list[dict[str, Any]] = []
    for backbone in sort_variants(backbone_frame["backbone"].dropna().unique().tolist()):
        variant_frame = backbone_frame[backbone_frame["backbone"] == backbone].copy()
        if variant_frame.empty:
            continue
        comparisons.append(
            {
                "backbone": backbone,
                "source_dataset": _source_dataset_name(dataset_frame, backbone),
                "table": _build_side_by_side_frame(original_frame, variant_frame, backbone),
            }
        )

    if not comparisons:
        return None

    backbone_mean = (
        backbone_frame.groupby("model", as_index=False)[["MAE", "RMSE", "WAPE"]].mean()
    )

    return {
        "base_dataset": base_dataset,
        "dataset_label": dataset_title,
        "original_dataset": base_dataset,
        "comparisons": comparisons,
        "mean_backbones": [comparison["backbone"] for comparison in comparisons],
        "mean_table": _build_side_by_side_frame(original_frame, backbone_mean, "Backbone Mean"),
    }


def _render_markdown_table(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in frame.to_dict("records"):
        cells: list[str] = []
        for column in columns:
            value = row.get(column)
            if column == "Model":
                cells.append(str(value))
            elif column == "Delta MAE (%)":
                cells.append(_format_delta(value, latex=False))
            else:
                cells.append(_format_metric(value))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _render_latex_table(frame: pd.DataFrame, *, caption: str, label: str) -> str:
    columns = frame.columns.tolist()
    column_spec = "l" + ("r" * (len(columns) - 1))
    header = " & ".join(_latex_escape(column) for column in columns) + r" \\"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for row in frame.to_dict("records"):
        cells: list[str] = []
        for column in columns:
            value = row.get(column)
            if column == "Model":
                cells.append(_latex_escape(value))
            elif column == "Delta MAE (%)":
                cells.append(_format_delta(value, latex=True))
            else:
                cells.append(_format_metric(value))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_markdown_dataset_report(bundle: dict[str, Any]) -> str:
    lines = [
        f"# {bundle['dataset_label']}",
        "",
        f"Dataset original: `{bundle['original_dataset']}`",
        "",
    ]

    for comparison in bundle["comparisons"]:
        lines.append(f"## Original vs {comparison['backbone']}")
        lines.append("")
        if comparison["source_dataset"]:
            lines.append(f"Dataset backbone: `{comparison['source_dataset']}`")
            lines.append("")
        lines.append(_render_markdown_table(comparison["table"]))
        lines.append("")

    mean_backbones = ", ".join(bundle["mean_backbones"])
    lines.append("## Original vs Media dos backbones")
    lines.append("")
    lines.append(f"Backbones considerados: `{mean_backbones}`")
    lines.append("")
    lines.append(_render_markdown_table(bundle["mean_table"]))
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_latex_dataset_report(bundle: dict[str, Any]) -> str:
    lines = [
        "% Requer \\usepackage{booktabs}",
        f"% Dataset: {bundle['dataset_label']}",
        "",
    ]

    for comparison in bundle["comparisons"]:
        caption = (
            f"{bundle['dataset_label']}: original ({bundle['original_dataset']}) vs "
            f"{comparison['backbone']} ({comparison['source_dataset'] or comparison['backbone']}). "
            "Delta calculado com base no MAE."
        )
        label = (
            f"tab:{_slug(bundle['base_dataset'])}-"
            f"original-vs-{_slug(comparison['backbone'])}"
        )
        lines.append(
            _render_latex_table(
                comparison["table"],
                caption=caption,
                label=label,
            )
        )

    mean_backbones = ", ".join(bundle["mean_backbones"])
    lines.append(
        _render_latex_table(
            bundle["mean_table"],
            caption=(
                f"{bundle['dataset_label']}: original vs media dos backbones "
                f"({mean_backbones}). Delta calculado com base no MAE."
            ),
            label=f"tab:{_slug(bundle['base_dataset'])}-original-vs-backbone-mean",
        )
    )

    return "\n".join(lines).rstrip() + "\n"


def write_dataset_comparison_reports(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    write_markdown: bool = True,
    write_latex: bool = True,
) -> dict[str, dict[str, str]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, str]] = {}
    for base_dataset in sorted(frame["base_dataset"].dropna().unique().tolist()):
        bundle = _build_dataset_bundle(frame, base_dataset)
        if bundle is None:
            continue

        dataset_outputs: dict[str, str] = {}
        filename = _dataset_filename(base_dataset)

        if write_markdown:
            md_path = output_path / f"{filename}.md"
            md_path.write_text(_render_markdown_dataset_report(bundle), encoding="utf-8")
            dataset_outputs["md"] = str(md_path)

        if write_latex:
            latex_path = output_path / f"{filename}.latex"
            latex_path.write_text(_render_latex_dataset_report(bundle), encoding="utf-8")
            dataset_outputs["latex"] = str(latex_path)

        outputs[bundle["dataset_label"]] = dataset_outputs

    return outputs


__all__ = [
    "BACKBONE_LABELS",
    "DATASET_LABELS",
    "DEFAULT_MODEL_ORDER",
    "METRIC_COLUMN_MAP",
    "VARIANT_ORDER",
    "backbone_method_name",
    "base_dataset_name",
    "dataset_label",
    "graph_variant",
    "load_family_results",
    "sort_models",
    "sort_variants",
    "write_dataset_comparison_reports",
]
