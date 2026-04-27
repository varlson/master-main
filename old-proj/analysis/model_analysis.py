#!/usr/bin/env python3
# encoding: utf-8

"""
Pipeline de sintese de resultados dos modelos.

O script descobre automaticamente os relatorios consolidados em:

    results/<mode>/md

e usa esses arquivos Markdown como ponto de entrada para localizar os CSVs
consolidados correspondentes em:

    results/<mode>/csv

Com base nisso, gera:
- arquivos por dataset com tabelas "original vs backbone"
- uma tabela final por dataset com "original vs media dos backbones"
- radar charts no esquema de comparacao definido para o artigo
- testes de Friedman e pos-teste de Nemenyi
- critical difference diagrams
- relatorio Markdown da analise

Exemplos:
    python3 analysis/model_analysis.py --mode both
    python3 analysis/model_analysis.py --mode original
    python3 analysis/model_analysis.py --mode backbone --metric RMSE
"""


from __future__ import annotations
# import sys
# import os
# sys.path.append(os.path.abspath(".."))

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.comparison_tables import (  # noqa: E402
    load_family_results,
    write_dataset_comparison_reports,
)
from shared.visualization import (  # noqa: E402
    RadarChart,
    friedman_test,
    nemenyi_test,
    plot_critical_difference_diagram,
)


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
METRIC_ALIASES = {
    "MAE": "MAE",
    "RMSE": "RMSE",
    "WAPE": "WAPE",
    "test_mae_mean": "MAE",
    "test_rmse_mean": "RMSE",
    "test_wape_mean": "WAPE",
}
REPORT_PATTERNS = {
    "original": [
        "original_all-datasets*_comparison_report.md",
        "all-datasets*_comparison_report.md",
    ],
    "backbone": [
        "backbone_all-datasets*_comparison_report.md",
    ],
}


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _results_mode_dir(results_root: Path, mode: str) -> Path:
    return results_root / mode


def _base_dataset_name(dataset: str) -> str:
    return str(dataset).split("-by-", 1)[0]


def _dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, str(dataset).upper())


def _backbone_method_name(dataset: str) -> str:
    dataset = str(dataset)
    match = re.search(r"-by-(.*?)-with-", dataset)
    if match:
        return match.group(1)
    if "-by-" in dataset:
        return dataset.split("-by-", 1)[1]
    return "original"


def _graph_variant(dataset: str, family: str) -> str:
    if family == "original":
        return "Full"
    return BACKBONE_LABELS.get(_backbone_method_name(dataset), _backbone_method_name(dataset))


def _normalize_metric_name(metric: str) -> str:
    normalized = METRIC_ALIASES.get(metric)
    if normalized is None:
        raise ValueError(
            f"Metrica '{metric}' nao suportada. Use uma entre: {sorted(METRIC_ALIASES)}"
        )
    return normalized


def _metric_column(metric: str) -> str:
    return METRIC_COLUMN_MAP[_normalize_metric_name(metric)]


def _dataset_slug(dataset_label: str) -> str:
    return dataset_label.lower().replace("-", "").replace(" ", "")


def _sort_models(models: list[str]) -> list[str]:
    preferred = {model: index for index, model in enumerate(DEFAULT_MODEL_ORDER)}
    return sorted(models, key=lambda model: (preferred.get(model, len(preferred)), model))


def _sort_variants(variants: list[str]) -> list[str]:
    return sorted(variants, key=lambda variant: (VARIANT_ORDER.get(variant, len(VARIANT_ORDER)), variant))


def _format_number(value: float, precision: int = 3) -> str:
    return f"{float(value):.{precision}f}"


def _format_delta(value: float) -> str:
    number = float(value)
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.1f}\\%"


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _report_prefix(md_path: Path) -> str:
    stem = md_path.stem
    for suffix in ["_comparison_report", "_selection_report"]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _resolve_csv_from_md(results_mode_dir: Path, md_path: Path) -> Path:
    csv_path = results_mode_dir / "csv" / f"{_report_prefix(md_path)}_consolidated_experiments.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV consolidado correspondente ao relatorio nao encontrado: {csv_path}"
        )
    return csv_path


def _matching_reports(md_dir: Path, family: str) -> list[Path]:
    candidates: list[Path] = []
    for pattern in REPORT_PATTERNS[family]:
        candidates.extend(md_dir.glob(pattern))

    unique_candidates: list[Path] = []
    for candidate in sorted(candidates):
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _select_report(md_dir: Path, family: str, requested: str | None = None) -> Path:
    candidates = _matching_reports(md_dir, family)
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum relatorio Markdown encontrado para '{family}' em {md_dir}."
        )

    if requested is None:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    requested_path = Path(requested)
    if requested_path.exists():
        return requested_path.resolve()

    project_requested_path = _resolve_project_path(requested)
    if project_requested_path.exists():
        return project_requested_path

    for candidate in candidates:
        if candidate.name == requested or candidate.stem == requested:
            return candidate
        if candidate.stem.startswith(requested):
            return candidate

    raise FileNotFoundError(
        f"Nao foi possivel localizar o relatorio '{requested}' em {md_dir}."
    )


def _analysis_tag(report_map: dict[str, Path]) -> str:
    normalized = []
    for path in report_map.values():
        prefix = _report_prefix(path)
        prefix = prefix.replace("original_", "", 1)
        prefix = prefix.replace("backbone_", "", 1)
        normalized.append(prefix)

    if not normalized:
        return f"models_{datetime.now().strftime('%d_%m_%Y-%Hh_%M_%S')}"

    if len(set(normalized)) == 1:
        return f"models_{normalized[0]}"

    compact = "__".join(sorted(set(normalized)))
    return f"models_{compact}"


def _load_family_frame(
    *,
    family: str,
    md_path: Path,
    csv_path: Path,
) -> pd.DataFrame:
    frame = load_family_results(csv_path, family)
    frame["source_md"] = str(md_path)
    frame["source_csv"] = str(csv_path)
    return frame


def _build_block_id(frame: pd.DataFrame, block_cols: list[str]) -> pd.Series:
    return frame[block_cols].astype(str).agg(" | ".join, axis=1)


def _latex_matrix(
    frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
    output_path: Path,
    float_precision: int = 4,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = frame.to_latex(
        index=True,
        escape=False,
        float_format=lambda value: f"{value:.{float_precision}f}",
        caption=caption,
        label=label,
    )
    output_path.write_text(latex, encoding="utf-8")
    return output_path


def _table1_dataset_frame(frame: pd.DataFrame, dataset_label: str) -> tuple[pd.DataFrame, list[str]]:
    dataset_frame = frame[frame["dataset_label"] == dataset_label].copy()
    backbones = _sort_variants(dataset_frame["backbone"].dropna().unique().tolist())
    rows: list[dict[str, Any]] = []

    for metric_name in ["MAE", "RMSE", "WAPE"]:
        for model in _sort_models(dataset_frame["model"].dropna().unique().tolist()):
            row: dict[str, Any] = {"Metric": metric_name, "Model": model}
            model_frame = dataset_frame[dataset_frame["model"] == model]
            for backbone in backbones:
                value = model_frame.loc[model_frame["backbone"] == backbone, metric_name].mean()
                row[backbone] = float(value) if pd.notna(value) else float("nan")
            rows.append(row)

    return pd.DataFrame(rows), backbones


def _table1_to_latex(table_frame: pd.DataFrame, dataset_label: str, backbones: list[str]) -> str:
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{Resultados absolutos em {dataset_label}.}}",
        f"\\label{{tab:table1_{_dataset_slug(dataset_label)}}}",
        "\\begin{tabular}{ll" + ("r" * len(backbones)) + "}",
        "\\toprule",
        "Metric & Model & " + " & ".join(backbones) + " \\\\",
        "\\midrule",
    ]

    for _, row in table_frame.iterrows():
        values = [row[backbone] for backbone in backbones if pd.notna(row[backbone])]
        best_value = min(values) if values else float("nan")
        cells = [str(row["Metric"]), str(row["Model"])]
        for backbone in backbones:
            value = row[backbone]
            if pd.isna(value):
                cells.append("--")
                continue
            text = _format_number(value)
            if abs(float(value) - float(best_value)) < 1e-12:
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _generate_table1(frame: pd.DataFrame, output_dir: Path) -> dict[str, dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str]] = {}

    for dataset_label in sorted(frame["dataset_label"].dropna().unique().tolist()):
        table_frame, backbones = _table1_dataset_frame(frame, dataset_label)
        if table_frame.empty:
            continue

        slug = _dataset_slug(dataset_label)
        csv_path = output_dir / f"table1_{slug}.csv"
        tex_path = output_dir / f"table1_{slug}.tex"

        table_frame.to_csv(csv_path, index=False)
        tex_path.write_text(
            _table1_to_latex(table_frame, dataset_label, backbones),
            encoding="utf-8",
        )
        outputs[dataset_label] = {"csv": str(csv_path), "tex": str(tex_path)}

    return outputs


def _table2_delta_frame(frame: pd.DataFrame, metric_name: str) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    backbones = [backbone for backbone in ["Full", "DF", "NC", "HSS"] if backbone in set(frame["backbone"])]
    if "Full" not in backbones:
        return pd.DataFrame(), backbones

    for dataset_label in sorted(frame["dataset_label"].dropna().unique().tolist()):
        dataset_frame = frame[frame["dataset_label"] == dataset_label]
        for model in _sort_models(dataset_frame["model"].dropna().unique().tolist()):
            model_frame = dataset_frame[dataset_frame["model"] == model]
            full_value = model_frame.loc[model_frame["backbone"] == "Full", metric_name].mean()
            if pd.isna(full_value):
                continue

            row: dict[str, Any] = {"Dataset": dataset_label, "Model": model}
            for backbone in backbones:
                if backbone == "Full":
                    row[backbone] = 0.0
                    continue
                value = model_frame.loc[model_frame["backbone"] == backbone, metric_name].mean()
                if pd.isna(value):
                    row[backbone] = float("nan")
                else:
                    row[backbone] = ((float(value) - float(full_value)) / float(full_value)) * 100.0
            rows.append(row)

    return pd.DataFrame(rows), backbones


def _table2_to_latex(table_frame: pd.DataFrame, metric_name: str, backbones: list[str]) -> str:
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{Delta percentual em {metric_name} em relacao ao Full.}}",
        f"\\label{{tab:table2_delta_{metric_name.lower()}}}",
        "\\begin{tabular}{ll" + ("r" * len(backbones)) + "}",
        "\\toprule",
        "Dataset & Model & " + " & ".join(backbones) + " \\\\",
        "\\midrule",
    ]

    comparison_backbones = [backbone for backbone in backbones if backbone != "Full"]
    for _, row in table_frame.iterrows():
        candidate_values = [
            float(row[backbone])
            for backbone in comparison_backbones
            if pd.notna(row[backbone])
        ]
        best_improvement = min(candidate_values) if candidate_values else None
        worst_degradation = max(candidate_values) if candidate_values else None

        cells = [str(row["Dataset"]), str(row["Model"])]
        for backbone in backbones:
            value = row[backbone]
            if pd.isna(value):
                cells.append("--")
                continue
            text = _format_delta(value)
            if backbone != "Full" and best_improvement is not None and abs(float(value) - best_improvement) < 1e-12:
                text = f"\\textbf{{{text}}}"
            if backbone != "Full" and worst_degradation is not None and abs(float(value) - worst_degradation) < 1e-12 and float(value) > 0:
                text = f"\\textcolor{{red}}{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _generate_table2(frame: pd.DataFrame, output_dir: Path, metric_name: str) -> dict[str, str] | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_frame, backbones = _table2_delta_frame(frame, metric_name)
    if table_frame.empty:
        return None

    slug = metric_name.lower()
    csv_path = output_dir / f"table2_delta_{slug}.csv"
    tex_path = output_dir / f"table2_delta_{slug}.tex"

    table_frame.to_csv(csv_path, index=False)
    tex_path.write_text(
        _table2_to_latex(table_frame, metric_name, backbones),
        encoding="utf-8",
    )
    return {"csv": str(csv_path), "tex": str(tex_path)}


def _radar_input_frame(frame: pd.DataFrame, label_col: str, label_order: list[str]) -> pd.DataFrame:
    radar_frame = frame.rename(
        columns={
            label_col: "model",
            "MAE": "test_mae_mean",
            "RMSE": "test_rmse_mean",
            "WAPE": "test_wape_mean",
        }
    )
    radar_frame["model"] = pd.Categorical(
        radar_frame["model"],
        categories=[label for label in label_order if label in set(radar_frame["model"])],
        ordered=True,
    )
    return radar_frame.sort_values("model")[["model", "test_mae_mean", "test_rmse_mean", "test_wape_mean"]]


def _generate_radar_models_full(frame: pd.DataFrame, output_dir: Path, normalize: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    full_frame = frame[frame["backbone"] == "Full"].copy()
    if full_frame.empty:
        return generated

    for dataset_label, dataset_frame in full_frame.groupby("dataset_label"):
        radar_frame = dataset_frame.groupby("model", as_index=False)[["MAE", "RMSE", "WAPE"]].mean()
        radar = RadarChart(normalize=normalize, model_order=_sort_models(radar_frame["model"].tolist()))
        output_path = output_dir / f"radar_models_{_dataset_slug(dataset_label)}_full.png"
        radar.plot(
            _radar_input_frame(radar_frame, "model", _sort_models(radar_frame["model"].tolist())),
            output_path=output_path,
            title=f"Perfil dos modelos - {dataset_label} (Full)",
        )
        generated.append(output_path)

    return generated


def _generate_radar_backbone_by_model(frame: pd.DataFrame, output_dir: Path, normalize: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    if frame.empty:
        return generated

    for model, model_frame in frame.groupby("model"):
        radar_frame = model_frame.groupby("backbone", as_index=False)[["MAE", "RMSE", "WAPE"]].mean()
        backbone_order = _sort_variants(radar_frame["backbone"].tolist())
        radar = RadarChart(normalize=normalize, model_order=backbone_order)
        output_path = output_dir / f"radar_backbone_{model}.png"
        radar.plot(
            _radar_input_frame(radar_frame, "backbone", backbone_order),
            output_path=output_path,
            title=f"{model} - Impacto do Backbone",
        )
        generated.append(output_path)

    return generated


def _generate_radar_backbones_by_dataset(frame: pd.DataFrame, output_dir: Path, normalize: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    if frame.empty:
        return generated

    for dataset_label, dataset_frame in frame.groupby("dataset_label"):
        radar_frame = dataset_frame.groupby("backbone", as_index=False)[["MAE", "RMSE", "WAPE"]].mean()
        backbone_order = _sort_variants(radar_frame["backbone"].tolist())
        radar = RadarChart(normalize=normalize, model_order=backbone_order)
        output_path = output_dir / f"radar_backbones_{_dataset_slug(dataset_label)}.png"
        radar.plot(
            _radar_input_frame(radar_frame, "backbone", backbone_order),
            output_path=output_path,
            title=f"Backbones - {dataset_label} (media dos modelos)",
        )
        generated.append(output_path)

    return generated


def _run_comparison_suite(
    *,
    frame: pd.DataFrame,
    comparison_name: str,
    unit_col: str,
    block_cols: list[str],
    metrics: list[str],
    stats_dir: Path,
    cd_dir: Path,
    alpha: float,
    cd_metrics: list[str] | None = None,
    cd_title_prefix: str | None = None,
) -> dict[str, Any] | None:
    if frame.empty:
        return None

    stats_dir.mkdir(parents=True, exist_ok=True)
    cd_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "comparison": comparison_name,
        "unit_col": unit_col,
        "block_cols": block_cols,
        "metrics": {},
    }

    for metric_name in metrics:
        metric_frame = frame[block_cols + [unit_col, metric_name]].dropna().copy()
        if metric_frame.empty:
            continue
        metric_frame["block_id"] = _build_block_id(metric_frame, block_cols)

        friedman = friedman_test(
            metric_frame,
            metric=metric_name,
            block_col="block_id",
            model_col=unit_col,
        )
        nemenyi = nemenyi_test(friedman["rank_matrix"], alpha=alpha)

        rank_csv = stats_dir / f"ranks_{comparison_name}_{metric_name}.csv"
        pairwise_csv = stats_dir / f"nemenyi_{comparison_name}_{metric_name}.csv"
        pvalues_csv = stats_dir / f"pvalues_{comparison_name}_{metric_name}.csv"
        tex_path = stats_dir / f"nemenyi_{comparison_name}_{metric_name}.tex"
        json_path = stats_dir / f"stats_{comparison_name}_{metric_name}.json"

        friedman["rank_matrix"].to_csv(rank_csv)
        nemenyi["pairwise"].to_csv(pairwise_csv, index=False)
        nemenyi["p_values"].to_csv(pvalues_csv)
        _latex_matrix(
            nemenyi["p_values"],
            caption=f"Nemenyi p-values - {comparison_name} - {metric_name}",
            label=f"tab:nemenyi_{comparison_name}_{metric_name.lower()}",
            output_path=tex_path,
        )

        metric_summary = {
            "friedman_statistic": float(friedman["statistic"]),
            "friedman_p_value": float(friedman["p_value"]),
            "n_blocks": int(friedman["n_blocks"]),
            "n_models": int(friedman["n_models"]),
            "critical_difference": float(nemenyi["critical_difference"]),
            "average_ranks": {
                label: float(rank)
                for label, rank in nemenyi["average_ranks"].items()
            },
            "files": {
                "rank_csv": str(rank_csv),
                "pairwise_csv": str(pairwise_csv),
                "pvalues_csv": str(pvalues_csv),
                "nemenyi_tex": str(tex_path),
            },
        }

        if cd_metrics and metric_name in cd_metrics:
            png_path = cd_dir / f"cd_{comparison_name}_{metric_name}.png"
            pdf_path = cd_dir / f"cd_{comparison_name}_{metric_name}.pdf"
            title = cd_title_prefix or comparison_name
            plot_critical_difference_diagram(
                average_ranks=nemenyi["average_ranks"],
                critical_difference=nemenyi["critical_difference"],
                pairwise=nemenyi["pairwise"],
                output_path=png_path,
                title=f"{title} ({metric_name})",
                alpha=alpha,
            )
            plot_critical_difference_diagram(
                average_ranks=nemenyi["average_ranks"],
                critical_difference=nemenyi["critical_difference"],
                pairwise=nemenyi["pairwise"],
                output_path=pdf_path,
                title=f"{title} ({metric_name})",
                alpha=alpha,
            )
            metric_summary["files"]["cd_png"] = str(png_path)
            metric_summary["files"]["cd_pdf"] = str(pdf_path)

        with json_path.open("w", encoding="utf-8") as file_obj:
            json.dump(metric_summary, file_obj, indent=2, ensure_ascii=False)
        metric_summary["files"]["summary_json"] = str(json_path)
        results["metrics"][metric_name] = metric_summary

    return results if results["metrics"] else None


def _write_analysis_report(
    *,
    output_path: Path,
    mode: str,
    primary_metric: str,
    alpha: float,
    normalize: str,
    source_reports: dict[str, Path],
    source_csvs: dict[str, Path],
    table_outputs: dict[str, Any],
    radar_outputs: dict[str, list[Path]],
    stats_outputs: dict[str, Any],
) -> None:
    def write_stats_suite(file_obj: Any, title: str, suite: dict[str, Any]) -> None:
        file_obj.write(f"### {title}\n\n")
        file_obj.write(f"- unit_col: `{suite['unit_col']}`\n")
        file_obj.write(f"- block_cols: `{suite['block_cols']}`\n\n")
        for metric_name, summary in suite["metrics"].items():
            file_obj.write(f"#### {metric_name}\n\n")
            file_obj.write(
                f"- Friedman: estatistica = `{summary['friedman_statistic']:.4f}`, "
                f"p-valor = `{summary['friedman_p_value']:.6f}`\n"
            )
            file_obj.write(
                f"- Critical Difference: `{summary['critical_difference']:.4f}`\n"
            )
            file_obj.write("- Average ranks:\n")
            for label, rank in summary["average_ranks"].items():
                file_obj.write(f"  - `{label}`: `{rank:.4f}`\n")
            file_obj.write("- Arquivos:\n")
            for _, path in summary["files"].items():
                file_obj.write(f"  - `{path}`\n")
            file_obj.write("\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("# Model Analysis Pipeline\n\n")
        file_obj.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file_obj.write(f"- Modo: `{mode}`\n")
        file_obj.write(f"- Metrica principal: `{primary_metric}`\n")
        file_obj.write(f"- Alpha: `{alpha}`\n")
        file_obj.write(f"- Normalizacao radar: `{normalize}`\n\n")

        file_obj.write("## Fontes\n\n")
        for family in sorted(source_reports):
            file_obj.write(f"- `{family}` md: `{source_reports[family]}`\n")
            file_obj.write(f"- `{family}` csv: `{source_csvs[family]}`\n")
        file_obj.write("\n")

        file_obj.write("## Tabelas\n\n")
        dataset_reports = table_outputs.get("dataset_reports", {})
        if dataset_reports:
            file_obj.write("### Comparacoes Original vs Backbone por dataset\n\n")
            for dataset_label, files in dataset_reports.items():
                if "md" in files:
                    file_obj.write(f"- `{dataset_label}` md: `{files['md']}`\n")
                if "latex" in files:
                    file_obj.write(f"- `{dataset_label}` latex: `{files['latex']}`\n")
            file_obj.write("\n")

        table1_outputs = table_outputs.get("table1", {})
        if table1_outputs:
            file_obj.write("### Tabela 1 - Resultados absolutos\n\n")
            for dataset_label, files in table1_outputs.items():
                file_obj.write(f"- `{dataset_label}` csv: `{files['csv']}`\n")
                file_obj.write(f"- `{dataset_label}` tex: `{files['tex']}`\n")
            file_obj.write("\n")

        table2_outputs = table_outputs.get("table2")
        if table2_outputs:
            file_obj.write("### Tabela 2 - Delta vs Full\n\n")
            file_obj.write(f"- csv: `{table2_outputs['csv']}`\n")
            file_obj.write(f"- tex: `{table2_outputs['tex']}`\n\n")

        file_obj.write("## Radar Charts\n\n")
        if not any(radar_outputs.values()):
            file_obj.write("Nenhum radar chart foi gerado.\n\n")
        else:
            for section_name, files in radar_outputs.items():
                if not files:
                    continue
                file_obj.write(f"### {section_name}\n\n")
                for path in files:
                    file_obj.write(f"- `{path}`\n")
                file_obj.write("\n")

        file_obj.write("## Testes Estatisticos\n\n")
        if not any(stats_outputs.values()):
            file_obj.write("Nenhum teste estatistico foi gerado.\n\n")
        else:
            for section_name, suite in stats_outputs.items():
                if not suite:
                    continue
                if isinstance(suite, dict) and "metrics" in suite and "unit_col" in suite:
                    write_stats_suite(file_obj, section_name, suite)
                    continue

                file_obj.write(f"### {section_name}\n\n")
                for nested_name, nested_suite in suite.items():
                    if not nested_suite:
                        continue
                    write_stats_suite(file_obj, nested_name, nested_suite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de analise dos modelos a partir de results/<mode>/md."
    )
    parser.add_argument(
        "--mode",
        choices=["original", "backbone", "both"],
        default="both",
        help="Subpasta em results/ a ser analisada.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Raiz de resultados. Ex.: results",
    )
    parser.add_argument(
        "--metric",
        default="test_mae_mean",
        help="Metrica principal para Friedman/Nemenyi.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha para o pos-teste de Nemenyi.",
    )
    parser.add_argument(
        "--normalize",
        choices=["relative_score", "minmax_inverse", "raw"],
        default="relative_score",
        help="Normalizacao usada no radar chart.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretorio de saida. Se omitido, usa results/<mode>/analysis/<tag>.",
    )
    parser.add_argument(
        "--original-report",
        default=None,
        help="Nome, prefixo ou caminho do relatorio md original.",
    )
    parser.add_argument(
        "--backbone-report",
        default=None,
        help="Nome, prefixo ou caminho do relatorio md backbone.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    results_root = _resolve_project_path(args.results_root)
    results_mode_dir = _results_mode_dir(results_root, args.mode)
    md_dir = results_mode_dir / "md"

    if not md_dir.exists():
        raise FileNotFoundError(f"Pasta de relatorios Markdown nao encontrada: {md_dir}")

    families = ["original", "backbone"] if args.mode == "both" else [args.mode]

    source_reports: dict[str, Path] = {}
    source_csvs: dict[str, Path] = {}
    frames: list[pd.DataFrame] = []

    requested_map = {
        "original": args.original_report,
        "backbone": args.backbone_report,
    }

    for family in families:
        md_path = _select_report(md_dir, family, requested=requested_map.get(family))
        csv_path = _resolve_csv_from_md(results_mode_dir, md_path)
        source_reports[family] = md_path
        source_csvs[family] = csv_path
        frames.append(
            _load_family_frame(
                family=family,
                md_path=md_path,
                csv_path=csv_path,
            )
        )

    if not frames:
        raise ValueError("Nenhum conjunto de resultados foi carregado para analise.")

    all_frame = pd.concat(frames, ignore_index=True)
    primary_metric = _normalize_metric_name(args.metric)

    if args.output_dir:
        output_root = _resolve_project_path(args.output_dir)
    else:
        output_root = results_mode_dir / "analysis" / _analysis_tag(source_reports)

    tables_dir = output_root / "tables"
    figures_radar_dir = output_root / "figures" / "radar"
    figures_cd_dir = output_root / "figures" / "cd"
    stats_dir = output_root / "stats"
    csv_dir = output_root / "csv"
    json_dir = output_root / "json"
    md_output_dir = output_root / "md"

    for path in [tables_dir, figures_radar_dir, figures_cd_dir, stats_dir, csv_dir, json_dir, md_output_dir]:
        path.mkdir(parents=True, exist_ok=True)

    loaded_results_csv = csv_dir / "loaded_results.csv"
    all_frame.to_csv(loaded_results_csv, index=False)

    metadata = {
        "mode": args.mode,
        "metric": primary_metric,
        "alpha": args.alpha,
        "normalize": args.normalize,
        "source_reports": {family: str(path) for family, path in source_reports.items()},
        "source_csvs": {family: str(path) for family, path in source_csvs.items()},
        "loaded_results_csv": str(loaded_results_csv),
    }
    metadata_json = json_dir / "analysis_metadata.json"
    with metadata_json.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=False)

    dataset_reports = write_dataset_comparison_reports(all_frame, tables_dir)
    table_outputs: dict[str, Any] = {
        "dataset_reports": dataset_reports,
    }
    if not dataset_reports:
        table_outputs["table1"] = _generate_table1(all_frame, tables_dir)
        table_outputs["table2"] = _generate_table2(all_frame, tables_dir, primary_metric)

    radar_outputs: dict[str, list[Path]] = {}
    radar_outputs["models_full_by_dataset"] = _generate_radar_models_full(
        all_frame,
        figures_radar_dir / "models_full",
        args.normalize,
    )
    radar_outputs["backbone_by_model"] = _generate_radar_backbone_by_model(
        all_frame,
        figures_radar_dir / "backbone_by_model",
        args.normalize,
    )
    radar_outputs["backbones_by_dataset"] = _generate_radar_backbones_by_dataset(
        all_frame,
        figures_radar_dir / "backbones_by_dataset",
        args.normalize,
    )

    stats_outputs: dict[str, Any] = {}
    metrics = ["MAE", "RMSE", "WAPE"]
    if len(set(all_frame["backbone"])) >= 2:
        stats_outputs["backbones_general"] = _run_comparison_suite(
            frame=all_frame.copy(),
            comparison_name="backbones",
            unit_col="backbone",
            block_cols=["dataset_label", "model"],
            metrics=metrics,
            stats_dir=stats_dir / "backbones_general",
            cd_dir=figures_cd_dir / "backbones_general",
            alpha=args.alpha,
            cd_metrics=[primary_metric],
            cd_title_prefix="Critical Difference - Configuracoes de grafo",
        )

    stats_outputs["models_general"] = _run_comparison_suite(
        frame=all_frame.copy(),
        comparison_name="models",
        unit_col="model",
        block_cols=["dataset_label", "backbone"],
        metrics=metrics,
        stats_dir=stats_dir / "models_general",
        cd_dir=figures_cd_dir / "models_general",
        alpha=args.alpha,
        cd_metrics=[],
        cd_title_prefix="Critical Difference - Modelos",
    )

    dataset_stats: dict[str, Any] = {}
    for dataset_label, dataset_frame in all_frame.groupby("dataset_label"):
        if len(set(dataset_frame["backbone"])) < 2:
            continue
        dataset_stats[dataset_label] = _run_comparison_suite(
            frame=dataset_frame.copy(),
            comparison_name=f"backbones_{_dataset_slug(dataset_label)}",
            unit_col="backbone",
            block_cols=["model"],
            metrics=metrics,
            stats_dir=stats_dir / "backbones_by_dataset" / _dataset_slug(dataset_label),
            cd_dir=figures_cd_dir / "backbones_by_dataset" / _dataset_slug(dataset_label),
            alpha=args.alpha,
            cd_metrics=metrics,
            cd_title_prefix=f"Critical Difference - {dataset_label} (n=3 blocos)",
        )
    if dataset_stats:
        stats_outputs["backbones_by_dataset"] = dataset_stats

    report_md = md_output_dir / "report.md"
    _write_analysis_report(
        output_path=report_md,
        mode=args.mode,
        primary_metric=primary_metric,
        alpha=args.alpha,
        normalize=args.normalize,
        source_reports=source_reports,
        source_csvs=source_csvs,
        table_outputs=table_outputs,
        radar_outputs=radar_outputs,
        stats_outputs=stats_outputs,
    )

    print(f"Analysis pipeline concluido. Resultados salvos em: {output_root}")
    print(f"- Metadata: {metadata_json}")
    print(f"- Report:   {report_md}")


if __name__ == "__main__":
    main()
