#!/usr/bin/env python3
# encoding: utf-8

"""
Gera arquivos LaTeX por dataset para comparacoes entre grafo original e backbones.

Uso:
    python3 shared/latex_table_scrpt.py

As tabelas geradas assumem uso de:
    \\usepackage{booktabs}
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.comparison_tables import (  # noqa: E402
    load_family_results,
    write_dataset_comparison_reports,
)


def _latest_consolidated_csv(results_root: str | Path, prefix: str) -> Path:
    csv_dir = Path(results_root) / "csv"
    candidates = sorted(csv_dir.glob(f"{prefix}*_consolidated_experiments.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum CSV consolidado encontrado em {csv_dir} com prefixo '{prefix}'."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_latex_tables(
    *,
    results_root: str | Path = "results/both",
    output_dir: str | Path | None = None,
    original_csv: str | Path | None = None,
    backbone_csv: str | Path | None = None,
) -> list[Path]:
    results_root = Path(results_root)
    out = Path(output_dir) if output_dir else results_root / "latex_tables"
    out.mkdir(parents=True, exist_ok=True)

    original_path = Path(original_csv) if original_csv else _latest_consolidated_csv(
        results_root, "original_all-datasets"
    )
    backbone_path = Path(backbone_csv) if backbone_csv else _latest_consolidated_csv(
        results_root, "backbone_all-datasets"
    )

    frame = pd.concat(
        [
            load_family_results(original_path, "original"),
            load_family_results(backbone_path, "backbone"),
        ],
        ignore_index=True,
    )

    outputs = write_dataset_comparison_reports(
        frame,
        out,
        write_markdown=False,
        write_latex=True,
    )

    generated = [
        Path(files["latex"])
        for _, files in sorted(outputs.items())
        if "latex" in files
    ]

    all_tables = ["% Requer \\usepackage{booktabs}", ""]
    for path in generated:
        all_tables.append(path.read_text(encoding="utf-8"))

    all_path = out / "all_tables.latex"
    all_path.write_text("\n".join(all_tables), encoding="utf-8")
    generated.append(all_path)
    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera tabelas LaTeX por dataset para original vs backbone."
    )
    parser.add_argument("--results-root", default="results/both")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--original-csv", default=None)
    parser.add_argument("--backbone-csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generated = write_latex_tables(
        results_root=args.results_root,
        output_dir=args.output_dir,
        original_csv=args.original_csv,
        backbone_csv=args.backbone_csv,
    )
    print("Tabelas LaTeX geradas:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
