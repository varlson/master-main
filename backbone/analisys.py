#!/usr/bin/env python3
# encoding: utf-8

"""Wrapper de compatibilidade para a analise de backbone via pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET_LIST = ["metr-la", "pems-bay"]
DEFAULT_METHODS = [
    "disp_fil",
    "nois_corr",
    "high_sal",
    "doub_stoch",
    "glanb",
    "h_backbone",
    "marg_likelihood",
]
DEFAULT_ALPHA = 0.1
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "analysis"


def _load_runtime():
    from pipeline.backbone.analysis_runtime import (
        main,
        run_analysis_pipeline,
        run_single_backbone_analysis,
    )

    return main, run_analysis_pipeline, run_single_backbone_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analisa redes originais e backbones para gerar tabelas e plots comparativos."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets a analisar no modo automatico.",
    )
    parser.add_argument(
        "--network-names",
        nargs="+",
        default=None,
        help="Lista opcional de nomes completos das redes backbone a considerar.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=DEFAULT_METHODS,
        help="Metodos usados para inferir nomes de backbone alpha.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha usado na inferencia dos nomes de backbone.",
    )
    parser.add_argument(
        "--original-path",
        default=None,
        help="Caminho explicito da rede original para analise customizada.",
    )
    parser.add_argument(
        "--backbone-paths",
        nargs="+",
        default=None,
        help="Caminhos explicitos dos backbones para analise customizada.",
    )
    parser.add_argument(
        "--dataset-label",
        default="custom",
        help="Label do dataset quando usar analise customizada.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Quantidade de hubs usados na comparacao de centralidade.",
    )
    parser.add_argument(
        "--robustness-steps",
        type=int,
        default=20,
        help="Numero de passos na curva de robustez.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=10,
        help="Numero de repeticoes para a robustez por remocao aleatoria.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Tag opcional para a pasta de saida.",
    )
    return parser


def run_analysis_pipeline(*args, **kwargs):
    _, runtime_run_analysis_pipeline, _ = _load_runtime()
    return runtime_run_analysis_pipeline(*args, **kwargs)


def run_single_backbone_analysis(*args, **kwargs):
    _, _, runtime_run_single_backbone_analysis = _load_runtime()
    return runtime_run_single_backbone_analysis(*args, **kwargs)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if any(flag in argv for flag in ("-h", "--help")):
        build_parser().parse_args(argv)
        return

    runtime_main, _, _ = _load_runtime()
    runtime_main(argv)


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_DATASET_LIST",
    "DEFAULT_METHODS",
    "OUTPUT_ROOT",
    "build_parser",
    "main",
    "run_analysis_pipeline",
    "run_single_backbone_analysis",
]


if __name__ == "__main__":
    main()
