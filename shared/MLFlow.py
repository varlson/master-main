from models.WaveNet import GraphWaveNet
from models.DCRNN import DCRNN
from models.MTGNN import MTGNN
from models.DGCRN import DGCRN
from models.STICformer import STICformer
from models.PatchSTG import PatchSTG
import torch
import mlflow
import mlflow.pytorch
from itertools import product
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
from shared.loaders import denormalize_predictions
from shared.visualization import generate_model_diagnostics


RESULTS_DIR: Path = Path("results")
CSV_DIR: Path = RESULTS_DIR / "csv"
JSON_DIR: Path = RESULTS_DIR / "json"
MD_DIR: Path = RESULTS_DIR / "md"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
BEST_MODELS_DIR: Path = RESULTS_DIR / "best-models"


def set_results_root(results_root: str | Path) -> None:
    global RESULTS_DIR, CSV_DIR, JSON_DIR, MD_DIR, PLOTS_DIR, BEST_MODELS_DIR

    RESULTS_DIR = Path(results_root)
    CSV_DIR = RESULTS_DIR / "csv"
    JSON_DIR = RESULTS_DIR / "json"
    MD_DIR = RESULTS_DIR / "md"
    PLOTS_DIR = RESULTS_DIR / "plots"
    BEST_MODELS_DIR = RESULTS_DIR / "best-models"

    for _dir in (RESULTS_DIR, CSV_DIR, JSON_DIR, MD_DIR, PLOTS_DIR, BEST_MODELS_DIR):
        _dir.mkdir(parents=True, exist_ok=True)


def _build_best_configs_dataframe(all_results):
    best_loss = min(all_results, key=lambda x: x['test_loss'])
    best_mae = min(all_results, key=lambda x: x['mae'])
    best_rmse = min(all_results, key=lambda x: x['rmse'])
    best_mape = min(all_results, key=lambda x: x['mape'])

    best_configs = {
        'Métrica': ['Loss', 'MAE', 'RMSE', 'MAPE'],
        'Test Loss': [best_loss['test_loss'], best_mae['test_loss'],
                     best_rmse['test_loss'], best_mape['test_loss']],
        'MAE': [best_loss['mae'], best_mae['mae'],
               best_rmse['mae'], best_mape['mae']],
        'RMSE': [best_loss['rmse'], best_mae['rmse'],
                best_rmse['rmse'], best_mape['rmse']],
        'MAPE (%)': [best_loss['mape'], best_mae['mape'],
                    best_rmse['mape'], best_mape['mape']],
        'Params': [str(best_loss['params']), str(best_mae['params']),
                  str(best_rmse['params']), str(best_mape['params'])]
    }
    df_best = pd.DataFrame(best_configs)
    return best_loss, best_mae, best_rmse, best_mape, df_best


def _save_grid_search_summary(experiment_name, model_name, all_results, best_loss, best_mae, best_rmse, best_mape):
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "model": model_name,
        "total_configs": len(all_results),
        "best_by_loss": best_loss,
        "best_by_mae": best_mae,
        "best_by_rmse": best_rmse,
        "best_by_mape": best_mape,
        "all_results": all_results
    }

    filename = JSON_DIR / f"{experiment_name}_summary.json"
    with open(filename, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n💾 Resumo completo salvo em: {filename}")


def _extract_dataset_name(experiment_name: str) -> str:
    parts = experiment_name.split("_")
    if len(parts) >= 4 and parts[0] in {"original", "backbone"}:
        return parts[1]
    if len(parts) >= 3:
        return parts[0]
    return "unknown"


def _configure_best_model_path(model, model_name: str, experiment_name: str) -> Path:
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "no-run-id"
    model_dir = BEST_MODELS_DIR / experiment_name / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / f"best_model_{model_name.lower()}.pth"
    model.best_model_path = str(best_model_path)
    return best_model_path


def _collect_predictions(model, test_loader):
    preds = model.predict(test_loader).detach().cpu().float()
    y_test = torch.cat([y for _, y in test_loader], dim=0).detach().cpu().float()
    return preds, y_test


def _compute_metrics(preds: torch.Tensor, y_test: torch.Tensor) -> tuple[float, float, float]:
    mae = torch.mean(torch.abs(preds - y_test)).item()
    rmse = torch.sqrt(torch.mean((preds - y_test) ** 2)).item()
    mape = (
        torch.mean(torch.abs((y_test - preds) / (torch.abs(y_test) + 1e-8))) * 100.0
    ).item()
    return mae, rmse, mape


def _to_numpy_for_visualization(
    preds: torch.Tensor, y_test: torch.Tensor, normalization_stats: dict | None
) -> tuple[np.ndarray, np.ndarray]:
    preds_np = preds.numpy()
    y_np = y_test.numpy()

    if normalization_stats:
        preds_np = denormalize_predictions(preds_np, normalization_stats)
        y_np = denormalize_predictions(y_np, normalization_stats)

    return preds_np, y_np


def _generate_and_log_plots(
    *,
    preds: torch.Tensor,
    y_test: torch.Tensor,
    model,
    model_name: str,
    experiment_name: str,
    normalization_stats: dict | None,
    generate_plots: bool,
    num_nodes_to_plot: int,
    max_time_points: int,
) -> None:
    if not generate_plots:
        return

    dataset_name = _extract_dataset_name(experiment_name)
    plot_dir = PLOTS_DIR / experiment_name
    train_losses = getattr(model, "train_losses", []) or []
    val_losses = getattr(model, "val_losses", []) or []

    preds_np, y_np = _to_numpy_for_visualization(preds, y_test, normalization_stats)

    report = generate_model_diagnostics(
        predictions=preds_np,
        targets=y_np,
        output_dir=plot_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        train_losses=train_losses,
        val_losses=val_losses,
        num_nodes_to_plot=num_nodes_to_plot,
        horizon_for_line_and_heatmap=0,
        max_points_line=max(150, max_time_points),
        max_time_points_heatmap=max_time_points,
        results_root=RESULTS_DIR,
    )
    mlflow.log_artifacts(str(plot_dir), artifact_path="plots")
    for file_path in report["generated_files"]:
        path = Path(file_path)
        if path.exists() and plot_dir not in path.parents:
            mlflow.log_artifact(str(path), artifact_path=path.parent.name)
    print(f"📈 Plots salvos em: {report['output_dir']}")


# ============================================
# GRAPH WAVENET - GRID SEARCH PADRONIZADO
# ============================================

def GraphWaveNet_train_with_mlflow(
    params,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="GraphWaveNet_Experiments",
    device='cpu',
    normalization_stats=None,
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
):
    """Treina modelo Graph WaveNet com tracking do MLflow"""
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log de hiperparâmetros
        mlflow.log_params(params)
        mlflow.log_param("device", device)
        mlflow.log_param("num_nodes", num_nodes)
        
        # Criar modelo Graph WaveNet
        model = GraphWaveNet(
            adj_mx=adj_mx,
            num_nodes=num_nodes,
            input_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=params['output_dim'],
            seq_len=params['seq_len'],
            horizon=params['horizon'],
            num_blocks=params.get('num_blocks', 4),
            dilation_base=params.get('dilation_base', 2),
            k=params.get('k', 2),
            dropout=params.get('dropout', 0.1),
            lr=params.get('lr', 1e-3),
            weight_decay=params.get('weight_decay', 1e-4),
            epochs=params.get('epochs', 50),
            patience=params.get('patience', 10),
            device=device
        )
        best_model_path = _configure_best_model_path(model, "GraphWaveNet", experiment_name)
        mlflow.log_param("best_model_path", str(best_model_path))
        
        # Treinar
        print("\nIniciando treinamento do Graph WaveNet...")
        model.fit(train_loader, val_loader)
        
        # Avaliar no teste
        test_loss = model.evaluate(test_loader)
        mlflow.log_metric("test_loss", test_loss)
        
        preds, y_test = _collect_predictions(model, test_loader)
        mae, rmse, mape = _compute_metrics(preds, y_test)
        _generate_and_log_plots(
            preds=preds,
            y_test=y_test,
            model=model,
            model_name="GraphWaveNet",
            experiment_name=experiment_name,
            normalization_stats=normalization_stats,
            generate_plots=generate_plots,
            num_nodes_to_plot=num_nodes_to_plot,
            max_time_points=max_time_points,
        )
        
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        
        # Salvar modelo no MLflow
        mlflow.pytorch.log_model(model, "graph_wavenet_model")
        
        return test_loss, mae, rmse, mape


def GraphWaveNet_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="GraphWaveNet_GridSearch",
    device='cpu',
    normalization_stats=None,
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
):
    """Executa grid search padronizado para Graph WaveNet"""
    
    keys = param_grid.keys()
    values = param_grid.values()
    
    all_results = []
    
    for combination in product(*values):
        params = dict(zip(keys, combination))
        
        print(f"\n{'='*60}")
        print(f"Testando combinação: {params}")
        print(f"{'='*60}")
        
        try:
            test_loss, mae, rmse, mape = GraphWaveNet_train_with_mlflow(
                params,
                train_loader,
                val_loader,
                test_loader,
                adj_mx,
                num_nodes,
                experiment_name=experiment_name,
                device=device,
                normalization_stats=normalization_stats,
                generate_plots=generate_plots,
                num_nodes_to_plot=num_nodes_to_plot,
                max_time_points=max_time_points,
            )
            
            all_results.append({
                'params': params,
                'test_loss': test_loss,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            })
            
            print(f"✅ Concluído: Loss={test_loss:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
            
        except Exception as e:
            # print(f"❌ Erro na combinação {params}: {e}")
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem: {str(e)}")
            print(f"\n{'='*60}")
            print("TRACEBACK COMPLETO:")
            print(f"{'='*60}")
            traceback.print_exc()
            continue
    
    # Processar e exibir resultados
    if all_results:
        
        # Encontrar melhor configuração por cada métrica
        best_loss = min(all_results, key=lambda x: x['test_loss'])
        best_mae = min(all_results, key=lambda x: x['mae'])
        best_rmse = min(all_results, key=lambda x: x['rmse'])
        best_mape = min(all_results, key=lambda x: x['mape'])
        
        # Criar DataFrame com todas as métricas
        print(f"\n{'='*80}")
        print("MELHORES CONFIGURAÇÕES POR MÉTRICA:")
        print(f"{'='*80}")
        
        best_configs = {
            'Métrica': ['Loss', 'MAE', 'RMSE', 'MAPE'],
            'Test Loss': [best_loss['test_loss'], best_mae['test_loss'], 
                         best_rmse['test_loss'], best_mape['test_loss']],
            'MAE': [best_loss['mae'], best_mae['mae'], 
                   best_rmse['mae'], best_mape['mae']],
            'RMSE': [best_loss['rmse'], best_mae['rmse'], 
                    best_rmse['rmse'], best_mape['rmse']],
            'MAPE (%)': [best_loss['mape'], best_mae['mape'], 
                        best_rmse['mape'], best_mape['mape']],
            'Params': [str(best_loss['params']), str(best_mae['params']),
                      str(best_rmse['params']), str(best_mape['params'])]
        }
        
        df_best = pd.DataFrame(best_configs)
        print("\n📊 RESUMO DAS MELHORES CONFIGURAÇÕES:\n")
        print(df_best.to_string(index=False))
        
        # Salvar resultados completos em JSON
        results_summary = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "model": "GraphWaveNet",
            "total_configs": len(all_results),
            "best_by_loss": best_loss,
            "best_by_mae": best_mae,
            "best_by_rmse": best_rmse,
            "best_by_mape": best_mape,
            "all_results": all_results
        }
        
        filename = JSON_DIR / f"{experiment_name}_summary.json"
        with open(filename, "w") as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n💾 Resumo completo salvo em: {filename}")
        print(f"{'='*80}\n")
        
        return all_results, best_mae, df_best
    else:
        print("❌ Nenhum resultado obtido!")
        return [], None, None


# ============================================
# DCRNN - GRID SEARCH PADRONIZADO
# ============================================

def DCRNN_train_with_mlflow(
    params,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="DCRNN_Traffic",
    device='cpu',
    normalization_stats=None,
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
):
    """Treina modelo DCRNN com tracking do MLflow"""
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log de hiperparâmetros
        mlflow.log_params(params)
        mlflow.log_param("device", device)
        mlflow.log_param("num_nodes", num_nodes)
        
        # Criar modelo
        

        model = DCRNN(
            adj_mx=adj_mx,
            num_nodes=num_nodes,
            input_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=params['output_dim'],
            seq_len=params['seq_len'],
            horizon=params['horizon'],
            k=params['k'],
            dropout=params['dropout'],
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            epochs=params['epochs'],
            patience=params['patience'],
            device=device,
            use_scheduled_sampling=params.get('use_scheduled_sampling', False),
            teacher_forcing_ratio=params.get('teacher_forcing_ratio', 0.5)
        )
        best_model_path = _configure_best_model_path(model, "DCRNN", experiment_name)
        mlflow.log_param("best_model_path", str(best_model_path))
        
        # Treinar
        print("\nIniciando treinamento do DCRNN...")
        model.fit(train_loader, val_loader)
        
        # Avaliar no teste
        test_loss = model.evaluate(test_loader)
        mlflow.log_metric("test_loss", test_loss)
        
        preds, y_test = _collect_predictions(model, test_loader)
        mae, rmse, mape = _compute_metrics(preds, y_test)
        _generate_and_log_plots(
            preds=preds,
            y_test=y_test,
            model=model,
            model_name="DCRNN",
            experiment_name=experiment_name,
            normalization_stats=normalization_stats,
            generate_plots=generate_plots,
            num_nodes_to_plot=num_nodes_to_plot,
            max_time_points=max_time_points,
        )
        
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        
        # Salvar modelo no MLflow
        mlflow.pytorch.log_model(model, "dcrnn_model")
        
        return test_loss, mae, rmse, mape


def DCRNN_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="DCRNN_GridSearch",
    device='cpu',
    normalization_stats=None,
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
):
    """Executa grid search padronizado para DCRNN"""
    
    keys = param_grid.keys()
    values = param_grid.values()
    
    all_results = []
    
    for combination in product(*values):
        params = dict(zip(keys, combination))
        
        print(f"\n{'='*60}")
        print(f"Testando combinação: {params}")
        print(f"{'='*60}")
        
        try:
            test_loss, mae, rmse, mape = DCRNN_train_with_mlflow(
                params,
                train_loader,
                val_loader,
                test_loader,
                adj_mx,
                num_nodes,
                experiment_name,
                device=device,
                normalization_stats=normalization_stats,
                generate_plots=generate_plots,
                num_nodes_to_plot=num_nodes_to_plot,
                max_time_points=max_time_points,
            )
            
            all_results.append({
                'params': params,
                'test_loss': test_loss,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            })
            
            print(f"✅ Concluído: Loss={test_loss:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
            
        except Exception as e:
            # print(f"❌ Erro na combinação {params}: {e}")
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem: {str(e)}")
            print(f"\n{'='*60}")
            print("TRACEBACK COMPLETO:")
            print(f"{'='*60}")
            traceback.print_exc()
            continue
    
    # Processar e exibir resultados
    if all_results:

        
        
        # Encontrar melhor configuração por cada métrica
        best_loss = min(all_results, key=lambda x: x['test_loss'])
        best_mae = min(all_results, key=lambda x: x['mae'])
        best_rmse = min(all_results, key=lambda x: x['rmse'])
        best_mape = min(all_results, key=lambda x: x['mape'])
        
        # Criar DataFrame com todas as métricas
        print(f"\n{'='*80}")
        print("MELHORES CONFIGURAÇÕES POR MÉTRICA:")
        print(f"{'='*80}")
        
        best_configs = {
            'Métrica': ['Loss', 'MAE', 'RMSE', 'MAPE'],
            'Test Loss': [best_loss['test_loss'], best_mae['test_loss'], 
                         best_rmse['test_loss'], best_mape['test_loss']],
            'MAE': [best_loss['mae'], best_mae['mae'], 
                   best_rmse['mae'], best_mape['mae']],
            'RMSE': [best_loss['rmse'], best_mae['rmse'], 
                    best_rmse['rmse'], best_mape['rmse']],
            'MAPE (%)': [best_loss['mape'], best_mae['mape'], 
                        best_rmse['mape'], best_mape['mape']],
            'Params': [str(best_loss['params']), str(best_mae['params']),
                      str(best_rmse['params']), str(best_mape['params'])]
        }
        
        df_best = pd.DataFrame(best_configs)
        print("\n📊 RESUMO DAS MELHORES CONFIGURAÇÕES:\n")
        print(df_best.to_string(index=False))
        
        # Salvar resultados completos em JSON
        results_summary = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "model": "DCRNN",
            "total_configs": len(all_results),
            "best_by_loss": best_loss,
            "best_by_mae": best_mae,
            "best_by_rmse": best_rmse,
            "best_by_mape": best_mape,
            "all_results": all_results
        }
        
        filename = JSON_DIR / f"{experiment_name}_summary.json"
        with open(filename, "w") as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n💾 Resumo completo salvo em: {filename}")
        print(f"{'='*80}\n")
        
        return all_results, best_mae, df_best
    else:
        print("❌ Nenhum resultado obtido!")
        return [], None, None


# ============================================
# MTGNN - GRID SEARCH PADRONIZADO
# ============================================

def MTGNN_train_with_mlflow(
    params,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="MTGNN_Traffic",
    device='cpu',
    normalization_stats=None,
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
):
    """Treina modelo MTGNN com tracking do MLflow"""

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("device", device)
        mlflow.log_param("num_nodes", num_nodes)

        model = MTGNN(
            adj_mx=adj_mx,
            num_nodes=num_nodes,
            input_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=params['output_dim'],
            seq_len=params['seq_len'],
            horizon=params['horizon'],
            num_blocks=params.get('num_blocks', 3),
            kernel_size=params.get('kernel_size', 2),
            dilation_base=params.get('dilation_base', 2),
            gcn_depth=params.get('gcn_depth', 2),
            propalpha=params.get('propalpha', 0.05),
            node_dim=params.get('node_dim', 16),
            dropout=params.get('dropout', 0.1),
            lr=params.get('lr', 1e-3),
            weight_decay=params.get('weight_decay', 1e-4),
            epochs=params.get('epochs', 50),
            patience=params.get('patience', 10),
            device=device
        )
        best_model_path = _configure_best_model_path(model, "MTGNN", experiment_name)
        mlflow.log_param("best_model_path", str(best_model_path))

        print("\nIniciando treinamento do MTGNN...")
        model.fit(train_loader, val_loader)

        test_loss = model.evaluate(test_loader)
        mlflow.log_metric("test_loss", test_loss)

        preds, y_test = _collect_predictions(model, test_loader)
        mae, rmse, mape = _compute_metrics(preds, y_test)
        _generate_and_log_plots(
            preds=preds,
            y_test=y_test,
            model=model,
            model_name="MTGNN",
            experiment_name=experiment_name,
            normalization_stats=normalization_stats,
            generate_plots=generate_plots,
            num_nodes_to_plot=num_nodes_to_plot,
            max_time_points=max_time_points,
        )

        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

        mlflow.pytorch.log_model(model, "mtgnn_model")

        return test_loss, mae, rmse, mape


def MTGNN_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="MTGNN_GridSearch",
    device='cpu',
    normalization_stats=None,
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
):
    """Executa grid search padronizado para MTGNN"""

    keys = param_grid.keys()
    values = param_grid.values()

    all_results = []

    for combination in product(*values):
        params = dict(zip(keys, combination))

        print(f"\n{'='*60}")
        print(f"Testando combinação: {params}")
        print(f"{'='*60}")

        try:
            test_loss, mae, rmse, mape = MTGNN_train_with_mlflow(
                params,
                train_loader,
                val_loader,
                test_loader,
                adj_mx,
                num_nodes,
                experiment_name=experiment_name,
                device=device,
                normalization_stats=normalization_stats,
                generate_plots=generate_plots,
                num_nodes_to_plot=num_nodes_to_plot,
                max_time_points=max_time_points,
            )

            all_results.append({
                'params': params,
                'test_loss': test_loss,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            })

            print(f"✅ Concluído: Loss={test_loss:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")

        except Exception as e:
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem: {str(e)}")
            print(f"\n{'='*60}")
            print("TRACEBACK COMPLETO:")
            print(f"{'='*60}")
            traceback.print_exc()
            continue

    if all_results:
        best_loss = min(all_results, key=lambda x: x['test_loss'])
        best_mae = min(all_results, key=lambda x: x['mae'])
        best_rmse = min(all_results, key=lambda x: x['rmse'])
        best_mape = min(all_results, key=lambda x: x['mape'])

        print(f"\n{'='*80}")
        print("MELHORES CONFIGURAÇÕES POR MÉTRICA:")
        print(f"{'='*80}")

        best_configs = {
            'Métrica': ['Loss', 'MAE', 'RMSE', 'MAPE'],
            'Test Loss': [best_loss['test_loss'], best_mae['test_loss'],
                         best_rmse['test_loss'], best_mape['test_loss']],
            'MAE': [best_loss['mae'], best_mae['mae'],
                   best_rmse['mae'], best_mape['mae']],
            'RMSE': [best_loss['rmse'], best_mae['rmse'],
                    best_rmse['rmse'], best_mape['rmse']],
            'MAPE (%)': [best_loss['mape'], best_mae['mape'],
                        best_rmse['mape'], best_mape['mape']],
            'Params': [str(best_loss['params']), str(best_mae['params']),
                      str(best_rmse['params']), str(best_mape['params'])]
        }

        df_best = pd.DataFrame(best_configs)
        print("\n📊 RESUMO DAS MELHORES CONFIGURAÇÕES:\n")
        print(df_best.to_string(index=False))

        results_summary = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "model": "MTGNN",
            "total_configs": len(all_results),
            "best_by_loss": best_loss,
            "best_by_mae": best_mae,
            "best_by_rmse": best_rmse,
            "best_by_mape": best_mape,
            "all_results": all_results
        }

        filename = JSON_DIR / f"{experiment_name}_summary.json"
        with open(filename, "w") as f:
            json.dump(results_summary, f, indent=2)

        print(f"\n💾 Resumo completo salvo em: {filename}")
        print(f"{'='*80}\n")

        return all_results, best_mae, df_best
    else:
        print("❌ Nenhum resultado obtido!")
        return [], None, None


# ============================================
# DGCRN - GRID SEARCH PADRONIZADO
# ============================================

def DGCRN_train_with_mlflow(
    params,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="DGCRN_Traffic",
    device='cpu',
    normalization_stats=None,
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
):
    """Treina modelo DGCRN com tracking do MLflow"""

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("device", device)
        mlflow.log_param("num_nodes", num_nodes)

        model = DGCRN(
            adj_mx=adj_mx,
            num_nodes=num_nodes,
            input_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=params['output_dim'],
            seq_len=params['seq_len'],
            horizon=params['horizon'],
            node_dim=params.get('node_dim', 16),
            gcn_depth=params.get('gcn_depth', 2),
            dropout=params.get('dropout', 0.1),
            lr=params.get('lr', 1e-3),
            weight_decay=params.get('weight_decay', 1e-4),
            epochs=params.get('epochs', 50),
            patience=params.get('patience', 10),
            device=device
        )
        best_model_path = _configure_best_model_path(model, "DGCRN", experiment_name)
        mlflow.log_param("best_model_path", str(best_model_path))

        print("\nIniciando treinamento do DGCRN...")
        model.fit(train_loader, val_loader)

        test_loss = model.evaluate(test_loader)
        mlflow.log_metric("test_loss", test_loss)

        preds, y_test = _collect_predictions(model, test_loader)
        mae, rmse, mape = _compute_metrics(preds, y_test)
        _generate_and_log_plots(
            preds=preds,
            y_test=y_test,
            model=model,
            model_name="DGCRN",
            experiment_name=experiment_name,
            normalization_stats=normalization_stats,
            generate_plots=generate_plots,
            num_nodes_to_plot=num_nodes_to_plot,
            max_time_points=max_time_points,
        )

        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

        mlflow.pytorch.log_model(model, "dgcrn_model")
        return test_loss, mae, rmse, mape


def DGCRN_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="DGCRN_GridSearch",
    device='cpu',
    normalization_stats=None,
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
):
    """Executa grid search padronizado para DGCRN"""

    keys = param_grid.keys()
    values = param_grid.values()
    all_results = []

    for combination in product(*values):
        params = dict(zip(keys, combination))

        print(f"\n{'='*60}")
        print(f"Testando combinação: {params}")
        print(f"{'='*60}")

        try:
            test_loss, mae, rmse, mape = DGCRN_train_with_mlflow(
                params,
                train_loader,
                val_loader,
                test_loader,
                adj_mx,
                num_nodes,
                experiment_name=experiment_name,
                device=device,
                normalization_stats=normalization_stats,
                generate_plots=generate_plots,
                num_nodes_to_plot=num_nodes_to_plot,
                max_time_points=max_time_points,
            )
            all_results.append({
                'params': params,
                'test_loss': test_loss,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            })
            print(f"✅ Concluído: Loss={test_loss:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        except Exception as e:
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem: {str(e)}")
            print(f"\n{'='*60}")
            print("TRACEBACK COMPLETO:")
            print(f"{'='*60}")
            traceback.print_exc()
            continue

    if all_results:
        best_loss, best_mae, best_rmse, best_mape, df_best = _build_best_configs_dataframe(all_results)
        print(f"\n{'='*80}")
        print("MELHORES CONFIGURAÇÕES POR MÉTRICA:")
        print(f"{'='*80}")
        print("\n📊 RESUMO DAS MELHORES CONFIGURAÇÕES:\n")
        print(df_best.to_string(index=False))

        _save_grid_search_summary(
            experiment_name=experiment_name,
            model_name="DGCRN",
            all_results=all_results,
            best_loss=best_loss,
            best_mae=best_mae,
            best_rmse=best_rmse,
            best_mape=best_mape,
        )

        print(f"{'='*80}\n")
        return all_results, best_mae, df_best
    else:
        print("❌ Nenhum resultado obtido!")
        return [], None, None


# ============================================
# STICFORMER - GRID SEARCH PADRONIZADO
# ============================================

def STICformer_train_with_mlflow(
    params,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="STICformer_Traffic",
    device='cpu',
    normalization_stats=None,
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
):
    """Treina modelo STICformer com tracking do MLflow"""

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("device", device)
        mlflow.log_param("num_nodes", num_nodes)

        model = STICformer(
            adj_mx=adj_mx,
            num_nodes=num_nodes,
            input_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=params['output_dim'],
            seq_len=params['seq_len'],
            horizon=params['horizon'],
            num_layers=params.get('num_layers', 2),
            num_heads=params.get('num_heads', 4),
            ff_multiplier=params.get('ff_multiplier', 2),
            dropout=params.get('dropout', 0.1),
            lr=params.get('lr', 1e-3),
            weight_decay=params.get('weight_decay', 1e-4),
            epochs=params.get('epochs', 50),
            patience=params.get('patience', 10),
            device=device
        )
        best_model_path = _configure_best_model_path(model, "STICformer", experiment_name)
        mlflow.log_param("best_model_path", str(best_model_path))

        print("\nIniciando treinamento do STICformer...")
        model.fit(train_loader, val_loader)

        test_loss = model.evaluate(test_loader)
        mlflow.log_metric("test_loss", test_loss)

        preds, y_test = _collect_predictions(model, test_loader)
        mae, rmse, mape = _compute_metrics(preds, y_test)
        _generate_and_log_plots(
            preds=preds,
            y_test=y_test,
            model=model,
            model_name="STICformer",
            experiment_name=experiment_name,
            normalization_stats=normalization_stats,
            generate_plots=generate_plots,
            num_nodes_to_plot=num_nodes_to_plot,
            max_time_points=max_time_points,
        )

        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

        mlflow.pytorch.log_model(model, "sticformer_model")
        return test_loss, mae, rmse, mape


def STICformer_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="STICformer_GridSearch",
    device='cpu',
    normalization_stats=None,
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
):
    """Executa grid search padronizado para STICformer"""

    keys = param_grid.keys()
    values = param_grid.values()
    all_results = []

    for combination in product(*values):
        params = dict(zip(keys, combination))

        print(f"\n{'='*60}")
        print(f"Testando combinação: {params}")
        print(f"{'='*60}")

        try:
            test_loss, mae, rmse, mape = STICformer_train_with_mlflow(
                params,
                train_loader,
                val_loader,
                test_loader,
                adj_mx,
                num_nodes,
                experiment_name=experiment_name,
                device=device,
                normalization_stats=normalization_stats,
                generate_plots=generate_plots,
                num_nodes_to_plot=num_nodes_to_plot,
                max_time_points=max_time_points,
            )
            all_results.append({
                'params': params,
                'test_loss': test_loss,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            })
            print(f"✅ Concluído: Loss={test_loss:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        except Exception as e:
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem: {str(e)}")
            print(f"\n{'='*60}")
            print("TRACEBACK COMPLETO:")
            print(f"{'='*60}")
            traceback.print_exc()
            continue

    if all_results:
        best_loss, best_mae, best_rmse, best_mape, df_best = _build_best_configs_dataframe(all_results)
        print(f"\n{'='*80}")
        print("MELHORES CONFIGURAÇÕES POR MÉTRICA:")
        print(f"{'='*80}")
        print("\n📊 RESUMO DAS MELHORES CONFIGURAÇÕES:\n")
        print(df_best.to_string(index=False))

        _save_grid_search_summary(
            experiment_name=experiment_name,
            model_name="STICformer",
            all_results=all_results,
            best_loss=best_loss,
            best_mae=best_mae,
            best_rmse=best_rmse,
            best_mape=best_mape,
        )

        print(f"{'='*80}\n")
        return all_results, best_mae, df_best
    else:
        print("❌ Nenhum resultado obtido!")
        return [], None, None


# ============================================
# PATCHSTG - GRID SEARCH PADRONIZADO
# ============================================

def PatchSTG_train_with_mlflow(
    params,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="PatchSTG_Traffic",
    device='cpu',
    normalization_stats=None,
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
):
    """Treina modelo PatchSTG com tracking do MLflow"""

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("device", device)
        mlflow.log_param("num_nodes", num_nodes)

        model = PatchSTG(
            adj_mx=adj_mx,
            num_nodes=num_nodes,
            input_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=params['output_dim'],
            seq_len=params['seq_len'],
            horizon=params['horizon'],
            patch_len=params.get('patch_len', 4),
            patch_stride=params.get('patch_stride', 2),
            num_layers=params.get('num_layers', 2),
            num_heads=params.get('num_heads', 4),
            ff_multiplier=params.get('ff_multiplier', 2),
            dropout=params.get('dropout', 0.1),
            lr=params.get('lr', 1e-3),
            weight_decay=params.get('weight_decay', 1e-4),
            epochs=params.get('epochs', 50),
            patience=params.get('patience', 10),
            device=device
        )
        best_model_path = _configure_best_model_path(model, "PatchSTG", experiment_name)
        mlflow.log_param("best_model_path", str(best_model_path))

        print("\nIniciando treinamento do PatchSTG...")
        model.fit(train_loader, val_loader)

        test_loss = model.evaluate(test_loader)
        mlflow.log_metric("test_loss", test_loss)

        preds, y_test = _collect_predictions(model, test_loader)
        mae, rmse, mape = _compute_metrics(preds, y_test)
        _generate_and_log_plots(
            preds=preds,
            y_test=y_test,
            model=model,
            model_name="PatchSTG",
            experiment_name=experiment_name,
            normalization_stats=normalization_stats,
            generate_plots=generate_plots,
            num_nodes_to_plot=num_nodes_to_plot,
            max_time_points=max_time_points,
        )

        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

        mlflow.pytorch.log_model(model, "patchstg_model")
        return test_loss, mae, rmse, mape


def PatchSTG_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="PatchSTG_GridSearch",
    device='cpu',
    normalization_stats=None,
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
):
    """Executa grid search padronizado para PatchSTG"""

    keys = param_grid.keys()
    values = param_grid.values()
    all_results = []

    for combination in product(*values):
        params = dict(zip(keys, combination))

        print(f"\n{'='*60}")
        print(f"Testando combinação: {params}")
        print(f"{'='*60}")

        try:
            test_loss, mae, rmse, mape = PatchSTG_train_with_mlflow(
                params,
                train_loader,
                val_loader,
                test_loader,
                adj_mx,
                num_nodes,
                experiment_name=experiment_name,
                device=device,
                normalization_stats=normalization_stats,
                generate_plots=generate_plots,
                num_nodes_to_plot=num_nodes_to_plot,
                max_time_points=max_time_points,
            )
            all_results.append({
                'params': params,
                'test_loss': test_loss,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            })
            print(f"✅ Concluído: Loss={test_loss:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        except Exception as e:
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem: {str(e)}")
            print(f"\n{'='*60}")
            print("TRACEBACK COMPLETO:")
            print(f"{'='*60}")
            traceback.print_exc()
            continue

    if all_results:
        best_loss, best_mae, best_rmse, best_mape, df_best = _build_best_configs_dataframe(all_results)
        print(f"\n{'='*80}")
        print("MELHORES CONFIGURAÇÕES POR MÉTRICA:")
        print(f"{'='*80}")
        print("\n📊 RESUMO DAS MELHORES CONFIGURAÇÕES:\n")
        print(df_best.to_string(index=False))

        _save_grid_search_summary(
            experiment_name=experiment_name,
            model_name="PatchSTG",
            all_results=all_results,
            best_loss=best_loss,
            best_mae=best_mae,
            best_rmse=best_rmse,
            best_mape=best_mape,
        )

        print(f"{'='*80}\n")
        return all_results, best_mae, df_best
    else:
        print("❌ Nenhum resultado obtido!")
        return [], None, None
