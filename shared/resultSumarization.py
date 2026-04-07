"""
Utilitarios para consolidacao cientifica de resultados de experimentos.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _results_subdir(save_path: Path, folder: str) -> Path:
    path = save_path / folder
    path.mkdir(parents=True, exist_ok=True)
    return path


def _derive_prefix_from_output(output_csv: str) -> str:
    suffix = "_consolidated_experiments.csv"
    if output_csv.endswith(suffix):
        return output_csv[: -len(suffix)]
    return Path(output_csv).stem


def _safe_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _flatten_records(experiments_data: List[Dict], key: str) -> pd.DataFrame:
    rows: list[dict] = []
    for exp in experiments_data:
        for record in exp.get(key, []) or []:
            row = {
                "experiment_name": exp["experiment_name"],
                "model": exp["model"],
                "dataset": exp["dataset"],
            }
            row.update(record)
            if "params" in row:
                row["params"] = _safe_json(row["params"])
            rows.append(row)
    return pd.DataFrame(rows)


def _flatten_config_summaries(experiments_data: List[Dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for exp in experiments_data:
        for summary in exp.get("config_summaries", []) or []:
            row = {
                "experiment_name": exp["experiment_name"],
                "model": exp["model"],
                "dataset": exp["dataset"],
                "params": _safe_json(summary.get("params", {})),
            }
            for key, value in summary.items():
                if key == "params":
                    continue
                row[key] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _export_long_tables(
    *,
    experiments_data: List[Dict],
    save_path: Path,
    prefix: str,
) -> None:
    csv_dir = _results_subdir(save_path, "csv")
    json_dir = _results_subdir(save_path, "json")

    tables = {
        "trial_results": _flatten_records(experiments_data, "trial_results"),
        "final_test_results": _flatten_records(experiments_data, "final_test_results"),
        "config_summaries": _flatten_config_summaries(experiments_data),
    }

    for name, frame in tables.items():
        csv_file = csv_dir / f"{prefix}_{name}.csv"
        json_file = json_dir / f"{prefix}_{name}.json"
        frame.to_csv(csv_file, index=False)
        with json_file.open("w", encoding="utf-8") as file:
            json.dump(frame.to_dict("records"), file, indent=2, ensure_ascii=False)
        print(f"✔ Tabela salva: {csv_file}")
        print(f"✔ JSON salvo: {json_file}")


def consolidate_experiment_results(
    experiments_data: List[Dict],
    output_csv: str = "consolidated_experiments.csv",
    output_json: str = "consolidated_experiments.json",
    primary_metric: str = "test_mae_mean",
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Consolida resultados finais de multiplos experimentos em formato cientifico.
    """
    if save_path is None:
        save_path = Path(".")
    else:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    csv_dir = _results_subdir(save_path, "csv")
    json_dir = _results_subdir(save_path, "json")
    prefix = _derive_prefix_from_output(output_csv)

    rows = []
    for exp in experiments_data:
        final_summary = exp.get("final_summary")
        if not final_summary:
            print(f"⚠️  Experimento {exp['experiment_name']} sem final_summary, pulando...")
            continue

        selected_config = exp.get("selected_config") or {}
        row = {
            "experiment_name": exp["experiment_name"],
            "model": exp["model"],
            "dataset": exp["dataset"],
            "selection_metric": final_summary.get("selection_metric"),
            "selected_params": _safe_json(final_summary.get("selected_params", {})),
            "selected_num_completed_seeds": final_summary.get("selected_num_completed_seeds"),
            "final_num_completed_seeds": final_summary.get("final_num_completed_seeds"),
            "timestamp": datetime.now().isoformat(),
            **{
                key: value
                for key, value in final_summary.items()
                if key.startswith("test_") and key != "test_loss_normalized"
            },
            **{
                key: value
                for key, value in final_summary.items()
                if key.startswith("test_loss_normalized_")
            },
            **{
                key: value
                for key, value in selected_config.items()
                if key.startswith("val_") or key == "num_completed_seeds"
            },
        }
        rows.append(row)

    if not rows:
        print("❌ Nenhum resultado final válido para consolidar!")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if primary_metric in df.columns:
        df["rank_in_dataset"] = (
            df.groupby("dataset")[primary_metric]
            .rank(method="dense", ascending=True)
            .astype(int)
        )
        best_per_dataset = df.groupby("dataset")[primary_metric].transform("min")
        df["delta_vs_best_pct"] = ((df[primary_metric] / best_per_dataset) - 1.0) * 100.0
        df = df.sort_values(["dataset", "rank_in_dataset", primary_metric, "model"]).reset_index(drop=True)

    csv_path = csv_dir / output_csv
    df.to_csv(csv_path, index=False)
    print(f"✔ CSV consolidado salvo: {csv_path}")

    detailed = {
        "timestamp": datetime.now().isoformat(),
        "primary_metric": primary_metric,
        "total_experiments": len(experiments_data),
        "successful_experiments": len(df),
        "summary_rows": df.to_dict("records"),
        "experiments": [
            {
                "experiment_name": exp["experiment_name"],
                "model": exp["model"],
                "dataset": exp["dataset"],
                "metadata": exp.get("metadata", {}),
                "selected_config": exp.get("selected_config"),
                "final_summary": exp.get("final_summary"),
            }
            for exp in experiments_data
        ],
    }

    json_path = json_dir / output_json
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(detailed, file, indent=2, ensure_ascii=False)
    print(f"✔ JSON consolidado salvo: {json_path}")

    _export_long_tables(
        experiments_data=experiments_data,
        save_path=save_path,
        prefix=prefix,
    )

    return df


def create_comparison_report(
    consolidated_df: pd.DataFrame,
    output_file: str = "comparison_report.md",
    save_path: Optional[Path] = None,
) -> None:
    """
    Cria relatório markdown com foco em comparacao cientifica.
    """
    if save_path is None:
        save_path = Path(".")
    else:
        save_path = Path(save_path)

    md_dir = _results_subdir(save_path, "md")
    report_path = md_dir / output_file

    model_summary = (
        consolidated_df.groupby("model", as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            avg_rank=("rank_in_dataset", "mean"),
            mean_mae=("test_mae_mean", "mean"),
            mean_rmse=("test_rmse_mean", "mean"),
            mean_wape=("test_wape_mean", "mean"),
        )
        .sort_values(["avg_rank", "mean_mae", "mean_rmse"])
    )

    with report_path.open("w", encoding="utf-8") as file:
        file.write("# Relatório de Comparação Científica de Experimentos\n\n")
        file.write(f"**Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write(f"**Total de experimentos consolidados:** {len(consolidated_df)}\n\n")
        file.write(
            "Os números abaixo representam desempenho final em teste para a configuração "
            "selecionada por validação, agregada por múltiplas seeds quando disponíveis.\n\n"
        )

        file.write("## Ranking por Dataset\n\n")
        for dataset in consolidated_df["dataset"].unique():
            dataset_df = consolidated_df[consolidated_df["dataset"] == dataset]
            file.write(f"### {dataset}\n\n")
            file.write("| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |\n")
            file.write("|------|--------|----------|-----------|---------------|----------------------|\n")
            for _, row in dataset_df.sort_values(["rank_in_dataset", "test_mae_mean"]).iterrows():
                file.write(
                    f"| {int(row['rank_in_dataset'])} | {row['model']} | "
                    f"{row['test_mae_mean']:.4f} ± {row.get('test_mae_std', 0.0):.4f} | "
                    f"{row['test_rmse_mean']:.4f} ± {row.get('test_rmse_std', 0.0):.4f} | "
                    f"{row.get('test_wape_mean', 0.0):.2f} ± {row.get('test_wape_std', 0.0):.2f} | "
                    f"{row.get('delta_vs_best_pct', 0.0):.2f} |\n"
                )
            file.write("\n")

        file.write("## Resumo por Modelo\n\n")
        file.write("| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |\n")
        file.write("|--------|----------|------------|-----------|------------|----------------|\n")
        for _, row in model_summary.iterrows():
            file.write(
                f"| {row['model']} | {int(row['datasets'])} | {row['avg_rank']:.2f} | "
                f"{row['mean_mae']:.4f} | {row['mean_rmse']:.4f} | {row['mean_wape']:.2f} |\n"
            )
        file.write("\n")

        file.write("## Configurações Selecionadas\n\n")
        for _, row in consolidated_df.iterrows():
            file.write(f"### {row['experiment_name']}\n\n")
            file.write(f"- Modelo: {row['model']}\n")
            file.write(f"- Dataset: {row['dataset']}\n")
            file.write(f"- Métrica de seleção: `{row['selection_metric']}`\n")
            file.write(f"- Seeds finais: {int(row.get('final_num_completed_seeds', 0))}\n")
            file.write(f"- Melhor validação (MAE): {row.get('val_mae_mean', float('nan')):.4f}\n")
            file.write(f"- Teste (MAE): {row.get('test_mae_mean', float('nan')):.4f}\n")
            file.write(f"- Parâmetros: `{row['selected_params']}`\n\n")

        file.write("## Tabela Completa\n\n")
        file.write(
            "| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | "
            "Test sMAPE (%) | Test WAPE (%) | Rank |\n"
        )
        file.write(
            "|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|\n"
        )
        for _, row in consolidated_df.iterrows():
            file.write(
                f"| {row['experiment_name']} | {row['model']} | {row['dataset']} | "
                f"{int(row.get('final_num_completed_seeds', 0))} | "
                f"{row.get('test_mae_mean', float('nan')):.4f} | "
                f"{row.get('test_rmse_mean', float('nan')):.4f} | "
                f"{row.get('test_smape_mean', float('nan')):.2f} | "
                f"{row.get('test_wape_mean', float('nan')):.2f} | "
                f"{int(row.get('rank_in_dataset', 0))} |\n"
            )

    print(f"✔ Relatório salvo: {report_path}")


def export_best_configs_to_json(
    consolidated_df: pd.DataFrame,
    output_file: str = "best_configs.json",
    save_path: Optional[Path] = None,
) -> None:
    """
    Exporta melhores configuracoes selecionadas por dataset/modelo.
    """
    if save_path is None:
        save_path = Path(".")
    else:
        save_path = Path(save_path)

    json_dir = _results_subdir(save_path, "json")
    best_configs: dict[str, dict[str, dict]] = {}

    for _, row in consolidated_df.iterrows():
        dataset = row["dataset"]
        model = row["model"]
        best_configs.setdefault(dataset, {})
        best_configs[dataset][model] = {
            "experiment_name": row["experiment_name"],
            "selection_metric": row["selection_metric"],
            "selected_params": json.loads(row["selected_params"]),
            "final_num_completed_seeds": int(row.get("final_num_completed_seeds", 0)),
            "test_mae_mean": float(row.get("test_mae_mean", float("nan"))),
            "test_mae_std": float(row.get("test_mae_std", float("nan"))),
            "test_rmse_mean": float(row.get("test_rmse_mean", float("nan"))),
            "test_rmse_std": float(row.get("test_rmse_std", float("nan"))),
            "test_wape_mean": float(row.get("test_wape_mean", float("nan"))),
            "rank_in_dataset": int(row.get("rank_in_dataset", 0)),
            "delta_vs_best_pct": float(row.get("delta_vs_best_pct", float("nan"))),
            "timestamp": row["timestamp"],
        }

    json_path = json_dir / output_file
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(best_configs, file, indent=2, ensure_ascii=False)

    print(f"✔ Melhores configurações salvas: {json_path}")


def analyze_hyperparameter_impact(
    experiments_data: List[Dict],
    model_name: str,
    output_file: str = "hyperparameter_analysis.csv",
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    if save_path is None:
        save_path = Path(".")
    else:
        save_path = Path(save_path)

    csv_dir = _results_subdir(save_path, "csv")
    trial_df = _flatten_records(experiments_data, "trial_results")
    if trial_df.empty:
        print("⚠️  Nenhum trial_result disponível para análise.")
        return pd.DataFrame()

    model_df = trial_df[trial_df["model"] == model_name].copy()
    if model_df.empty:
        print(f"⚠️  Nenhum experimento encontrado para o modelo '{model_name}'")
        return pd.DataFrame()

    grouped = (
        model_df.groupby("params", as_index=False)
        .agg(
            completed_seeds=("seed", "nunique"),
            val_mae_mean=("val_mae", "mean"),
            val_rmse_mean=("val_rmse", "mean"),
            val_wape_mean=("val_wape", "mean"),
        )
        .sort_values(["val_mae_mean", "val_rmse_mean"])
    )

    output_path = csv_dir / output_file
    grouped.to_csv(output_path, index=False)
    print(f"✔ Análise de hiperparâmetros salva: {output_path}")
    return grouped
