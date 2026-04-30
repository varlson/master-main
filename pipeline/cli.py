from __future__ import annotations

import argparse

COMMAND_FORECAST = "forecast"
COMMAND_BUILD_BACKBONES = "build-backbones"
COMMAND_ANALYZE_BACKBONES = "analyze-backbones"
SUPPORTED_COMMANDS = (
    COMMAND_FORECAST,
    COMMAND_BUILD_BACKBONES,
    COMMAND_ANALYZE_BACKBONES,
)
BACKBONE_METHODS = (
    "disp_fil",
    "nois_corr",
    "high_sal",
    "doub_stoch",
    "glanb",
    "h_backbone",
    "marg_likelihood",
)
ANALYSIS_METHODS = (
    "disp_fil",
    "nois_corr",
    "high_sal",
    "doub_stoch",
    "glanb",
    "h_backbone",
    "marg_likelihood",
)


def _normalize_argv(argv: list[str] | None) -> list[str]:
    normalized = list(argv or [])
    if not normalized:
        return [COMMAND_FORECAST]
    if normalized[0] in {"-h", "--help"}:
        return normalized
    if normalized[0] in SUPPORTED_COMMANDS:
        return normalized
    return [COMMAND_FORECAST, *normalized]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrada principal unificada do projeto pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    forecast_parser = subparsers.add_parser(
        COMMAND_FORECAST,
        help="Executa o pipeline principal de forecasting.",
    )
    forecast_parser.add_argument(
        "--config",
        default="configs/search/pipeline.search.example.json",
        help="Caminho para o arquivo JSON de configuracao.",
    )
    forecast_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida a configuracao e imprime o plano de execucao sem treinar modelos.",
    )

    build_backbones_parser = subparsers.add_parser(
        COMMAND_BUILD_BACKBONES,
        help="Gera redes backbone e artefatos derivados.",
    )
    build_backbones_parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Lista de datasets a processar.",
    )
    build_backbones_parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(BACKBONE_METHODS),
        default=list(BACKBONE_METHODS),
        help="Metodos de backbone a executar.",
    )
    build_backbones_parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Threshold de alpha usado na poda.",
    )
    build_backbones_parser.add_argument(
        "--min-degree",
        type=int,
        default=None,
        help="Grau minimo de no.",
    )

    analyze_backbones_parser = subparsers.add_parser(
        COMMAND_ANALYZE_BACKBONES,
        help="Gera tabelas, plots e relatorios comparativos dos backbones.",
    )
    analyze_backbones_parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets a analisar no modo automatico.",
    )
    analyze_backbones_parser.add_argument(
        "--network-names",
        nargs="+",
        default=None,
        help="Lista opcional de nomes completos das redes backbone a considerar.",
    )
    analyze_backbones_parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(ANALYSIS_METHODS),
        default=list(ANALYSIS_METHODS),
        help="Metodos usados para inferir nomes de backbone alpha.",
    )
    analyze_backbones_parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha usado na inferencia dos nomes de backbone.",
    )
    analyze_backbones_parser.add_argument(
        "--original-path",
        default=None,
        help="Caminho explicito da rede original para analise customizada.",
    )
    analyze_backbones_parser.add_argument(
        "--backbone-paths",
        nargs="+",
        default=None,
        help="Caminhos explicitos dos backbones para analise customizada.",
    )
    analyze_backbones_parser.add_argument(
        "--dataset-label",
        default="custom",
        help="Label do dataset quando usar analise customizada.",
    )
    analyze_backbones_parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Quantidade de hubs usados na comparacao de centralidade.",
    )
    analyze_backbones_parser.add_argument(
        "--robustness-steps",
        type=int,
        default=20,
        help="Numero de passos na curva de robustez.",
    )
    analyze_backbones_parser.add_argument(
        "--random-trials",
        type=int,
        default=10,
        help="Numero de repeticoes para a robustez por remocao aleatoria.",
    )
    analyze_backbones_parser.add_argument(
        "--output-tag",
        default=None,
        help="Tag opcional para a pasta de saida.",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(_normalize_argv(argv))
