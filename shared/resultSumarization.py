"""
Utilitários para consolidação de resultados de experimentos
"""

import pandas as pd
from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Optional


def consolidate_experiment_results(
    experiments_data: List[Dict],
    output_csv: str = "consolidated_experiments.csv",
    output_json: str = "consolidated_experiments.json",
    primary_metric: str = "MAE",
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Consolida resultados de múltiplos experimentos
    
    Args:
        experiments_data: Lista de dicionários com dados dos experimentos
        output_csv: Nome do arquivo CSV de saída
        output_json: Nome do arquivo JSON de saída
        primary_metric: Métrica principal para consolidação
        save_path: Diretório onde salvar os resultados (None = diretório atual)
        
    Returns:
        DataFrame consolidado com todos os resultados
    """
    if save_path is None:
        save_path = Path(".")
    else:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
    
    rows = []
    
    for exp in experiments_data:
        df_best = exp["df_best"]
        
        if df_best is None or df_best.empty:
            print(f"⚠️  Experimento {exp['experiment_name']} não tem resultados válidos, pulando...")
            continue
        
        # Seleciona apenas a métrica principal
        primary_rows = df_best[df_best["Métrica"] == primary_metric]
        
        if primary_rows.empty:
            print(f"⚠️  Métrica '{primary_metric}' não encontrada em {exp['experiment_name']}, pulando...")
            continue
            
        row = primary_rows.iloc[0]
        
        rows.append({
            "experiment_name": exp["experiment_name"],
            "model": exp["model"],
            "dataset": exp["dataset"],
            "test_loss": row["Test Loss"],
            "mae": row["MAE"],
            "rmse": row["RMSE"],
            "mape": row["MAPE (%)"],
            "params": row["Params"],
            **{f"metadata_{k}": v for k, v in exp.get("metadata", {}).items()},
            "timestamp": datetime.now().isoformat()
        })
    
    if not rows:
        print("❌ Nenhum resultado válido para consolidar!")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Salvar CSV
    csv_path = save_path / output_csv
    df.to_csv(csv_path, index=False)
    print(f"✔ CSV salvo: {csv_path}")
    
    # JSON completo contendo todas as métricas
    detailed = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(experiments_data),
        "successful_experiments": len(rows),
        "primary_metric": primary_metric,
        "experiments": [
            {
                "experiment_name": exp["experiment_name"],
                "model": exp["model"],
                "dataset": exp["dataset"],
                "metadata": exp.get("metadata", {}),
                "best_results": exp["df_best"].to_dict("records") if exp["df_best"] is not None else []
            }
            for exp in experiments_data
        ]
    }
    
    json_path = save_path / output_json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    
    print(f"✔ JSON salvo: {json_path}")
    
    return df


def create_comparison_report(
    consolidated_df: pd.DataFrame,
    output_file: str = "comparison_report.md",
    save_path: Optional[Path] = None
) -> None:
    """
    Cria relatório markdown comparando todos os experimentos
    
    Args:
        consolidated_df: DataFrame consolidado
        output_file: Nome do arquivo de saída
        save_path: Diretório onde salvar
    """
    if save_path is None:
        save_path = Path(".")
    else:
        save_path = Path(save_path)
    
    report_path = save_path / output_file
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Comparação de Experimentos GNN\n\n")
        f.write(f"**Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total de experimentos:** {len(consolidated_df)}\n\n")
        
        # Sumário por modelo
        f.write("## Resumo por Modelo\n\n")
        for model in consolidated_df['model'].unique():
            model_df = consolidated_df[consolidated_df['model'] == model]
            f.write(f"### {model}\n\n")
            f.write(f"- Experimentos: {len(model_df)}\n")
            f.write(f"- MAE médio: {model_df['mae'].mean():.4f} ± {model_df['mae'].std():.4f}\n")
            f.write(f"- RMSE médio: {model_df['rmse'].mean():.4f} ± {model_df['rmse'].std():.4f}\n")
            f.write(f"- MAPE médio: {model_df['mape'].mean():.2f}% ± {model_df['mape'].std():.2f}%\n\n")
        
        # Melhores resultados por dataset
        f.write("## Melhores Resultados por Dataset\n\n")
        for dataset in consolidated_df['dataset'].unique():
            dataset_df = consolidated_df[consolidated_df['dataset'] == dataset]
            best_exp = dataset_df.loc[dataset_df['mae'].idxmin()]
            
            f.write(f"### {dataset}\n\n")
            f.write(f"- **Melhor modelo:** {best_exp['model']}\n")
            f.write(f"- **Experimento:** {best_exp['experiment_name']}\n")
            f.write(f"- **MAE:** {best_exp['mae']:.4f}\n")
            f.write(f"- **RMSE:** {best_exp['rmse']:.4f}\n")
            f.write(f"- **MAPE:** {best_exp['mape']:.2f}%\n")
            f.write(f"- **Parâmetros:** `{best_exp['params']}`\n\n")
        
        # Tabela completa
        f.write("## Tabela Completa de Resultados\n\n")
        f.write("| Experimento | Modelo | Dataset | MAE | RMSE | MAPE (%) |\n")
        f.write("|-------------|--------|---------|-----|------|----------|\n")
        
        for _, row in consolidated_df.iterrows():
            f.write(f"| {row['experiment_name']} | {row['model']} | {row['dataset']} | "
                   f"{row['mae']:.4f} | {row['rmse']:.4f} | {row['mape']:.2f} |\n")
    
    print(f"✔ Relatório salvo: {report_path}")


def export_best_configs_to_json(
    consolidated_df: pd.DataFrame,
    output_file: str = "best_configs.json",
    save_path: Optional[Path] = None
) -> None:
    """
    Exporta as melhores configurações por dataset/modelo para JSON
    
    Args:
        consolidated_df: DataFrame consolidado
        output_file: Nome do arquivo de saída
        save_path: Diretório onde salvar
    """
    if save_path is None:
        save_path = Path(".")
    else:
        save_path = Path(save_path)
    
    best_configs = {}
    
    for dataset in consolidated_df['dataset'].unique():
        best_configs[dataset] = {}
        
        for model in consolidated_df['model'].unique():
            subset = consolidated_df[
                (consolidated_df['dataset'] == dataset) &
                (consolidated_df['model'] == model)
            ]
            
            if len(subset) > 0:
                best_exp = subset.loc[subset['mae'].idxmin()]
                
                best_configs[dataset][model] = {
                    "experiment_name": best_exp['experiment_name'],
                    "mae": float(best_exp['mae']),
                    "rmse": float(best_exp['rmse']),
                    "mape": float(best_exp['mape']),
                    "test_loss": float(best_exp['test_loss']),
                    "params": best_exp['params'],
                    "timestamp": best_exp['timestamp']
                }
    
    json_path = save_path / output_file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(best_configs, f, indent=2, ensure_ascii=False)
    
    print(f"✔ Melhores configurações salvas: {json_path}")


def analyze_hyperparameter_impact(
    experiments_data: List[Dict],
    model_name: str,
    output_file: str = "hyperparameter_analysis.csv",
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Analisa o impacto de hiperparâmetros nos resultados
    
    Args:
        experiments_data: Lista de experimentos
        model_name: Nome do modelo a analisar
        output_file: Nome do arquivo de saída
        save_path: Diretório onde salvar
        
    Returns:
        DataFrame com análise de hiperparâmetros
    """
    if save_path is None:
        save_path = Path(".")
    else:
        save_path = Path(save_path)
    
    # Filtrar experimentos do modelo específico
    model_exps = [exp for exp in experiments_data if exp['model'] == model_name]
    
    if not model_exps:
        print(f"⚠️  Nenhum experimento encontrado para o modelo '{model_name}'")
        return pd.DataFrame()
    
    # TODO: Implementar análise mais sofisticada de hiperparâmetros
    # Esta é uma implementação básica
    
    print(f"ℹ️  Análise de hiperparâmetros para '{model_name}' não implementada completamente")
    return pd.DataFrame()