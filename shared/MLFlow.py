from models.WaveNet import GraphWaveNet
from models.DCRNN import DCRNN
from models.MTGNN import MTGNN
import torch
import mlflow
import mlflow.pytorch
from itertools import product
import json
from datetime import datetime
import pandas as pd
import traceback
import sys


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
    device='cpu'
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
        
        # Treinar
        print("\nIniciando treinamento do Graph WaveNet...")
        model.fit(train_loader, val_loader)
        
        # Avaliar no teste
        test_loss = model.evaluate(test_loader)
        mlflow.log_metric("test_loss", test_loss)
        
        # Calcular métricas adicionais
        model.eval()
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for X, Y in test_loader:
                X, Y = X.to(device), Y.to(device)
                Y_pred = model.forward(X)
                predictions.append(Y_pred.cpu())
                ground_truth.append(Y.cpu())
        
        predictions = torch.cat(predictions, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)
        
        mae = torch.mean(torch.abs(predictions - ground_truth)).item()
        rmse = torch.sqrt(torch.mean((predictions - ground_truth) ** 2)).item()
        mape = torch.mean(torch.abs((ground_truth - predictions) / (ground_truth + 1e-8))) * 100
        
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape.item())
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        
        # Salvar modelo no MLflow
        mlflow.pytorch.log_model(model, "graph_wavenet_model")
        
        return test_loss, mae, rmse, mape.item()


def GraphWaveNet_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="GraphWaveNet_GridSearch",
    device='cpu'
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
                device=device
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
        
        filename = f"results/{experiment_name}_summary.json"
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
    device='cpu'
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
        
        # Treinar
        print("\nIniciando treinamento do DCRNN...")
        model.fit(train_loader, val_loader)
        
        # Avaliar no teste
        test_loss = model.evaluate(test_loader)
        mlflow.log_metric("test_loss", test_loss)
        
        # Calcular métricas adicionais
        preds = model.predict(test_loader)
        Y_test = torch.cat([y for _, y in test_loader], dim=0)
        
        mae = torch.mean(torch.abs(preds - Y_test)).item()
        rmse = torch.sqrt(torch.mean((preds - Y_test) ** 2)).item()
        mape = torch.mean(torch.abs((Y_test - preds) / (Y_test + 1e-8))) * 100
        
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape.item())
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        
        # Salvar modelo no MLflow
        mlflow.pytorch.log_model(model, "dcrnn_model")
        
        return test_loss, mae, rmse, mape.item()


def DCRNN_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="DCRNN_GridSearch",
    device='cpu'
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
                device=device
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
        
        filename = f"results/{experiment_name}_summary.json"
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
    device='cpu'
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

        print("\nIniciando treinamento do MTGNN...")
        model.fit(train_loader, val_loader)

        test_loss = model.evaluate(test_loader)
        mlflow.log_metric("test_loss", test_loss)

        preds = model.predict(test_loader)
        Y_test = torch.cat([y for _, y in test_loader], dim=0)

        mae = torch.mean(torch.abs(preds - Y_test)).item()
        rmse = torch.sqrt(torch.mean((preds - Y_test) ** 2)).item()
        mape = torch.mean(torch.abs((Y_test - preds) / (Y_test + 1e-8))) * 100

        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape.item())

        print(f"Test Loss: {test_loss:.4f}")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

        mlflow.pytorch.log_model(model, "mtgnn_model")

        return test_loss, mae, rmse, mape.item()


def MTGNN_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="MTGNN_GridSearch",
    device='cpu'
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
                device=device
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

        filename = f"results/{experiment_name}_summary.json"
        with open(filename, "w") as f:
            json.dump(results_summary, f, indent=2)

        print(f"\n💾 Resumo completo salvo em: {filename}")
        print(f"{'='*80}\n")

        return all_results, best_mae, df_best
    else:
        print("❌ Nenhum resultado obtido!")
        return [], None, None
