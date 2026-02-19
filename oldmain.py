"""
Pipeline Principal de Experimentos GNN para Previsão de Tráfego

Este script executa experimentos completos com múltiplos modelos GNN,
datasets e configurações de backbone.

Uso:
    python main.py --models DCRNN GraphWaveNet --datasets metr-la pems-bay
    python main.py --quick-test  # Modo rápido para debug
    python main.py --backbone-only  # Apenas experimentos com backbone
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import torch
import mlflow

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent))
from config import *

from shared.loaders import prepare_dataloaders, load_graphml_backbone
from shared.resultSumarization import (
    consolidate_experiment_results,
    create_comparison_report,
    export_best_configs_to_json
)
from shared.models_registry import model_registry, register_default_models

# Importar grid search functions
from shared.MLFlow import DCRNN_grid_search, GraphWaveNet_grid_search


def parse_arguments():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description="Pipeline de experimentos GNN para previsão de tráfego"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Modelos a executar (ex: DCRNN GraphWaveNet). Se não especificado, executa todos."
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets a usar (ex: metr-la pems-bay). Se não especificado, usa todos."
    )
    
    parser.add_argument(
        "--skip-original",
        action="store_true",
        help="Pular experimentos com dados originais (sem backbone)"
    )
    
    parser.add_argument(
        "--backbone-only",
        action="store_true",
        help="Executar apenas experimentos com backbone"
    )
    
    parser.add_argument(
        "--backbone-methods",
        nargs="+",
        default=None,
        help="Métodos de backbone a usar (ex: disparity noise threshold)"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Modo de teste rápido (reduz épocas e grid search)"
    )
    
    parser.add_argument(
        "--device",
        default=None,
        help="Dispositivo a usar (cpu, cuda, cuda:0, etc.)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Diretório para salvar resultados"
    )
    
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help="URI de tracking do MLflow"
    )
    
    return parser.parse_args()


def setup_experiment(args):
    """
    Configura o ambiente de experimento
    
    Args:
        args: Argumentos parseados
        
    Returns:
        Dicionário com configurações
    """
    # Device
    if args.device:
        device = args.device
    else:
        device = DEVICE
    
    # Quick test mode
    if args.quick_test:
        print("\n⚡ MODO DE TESTE RÁPIDO ATIVADO")
        print("   - Épocas reduzidas para 2")
        print("   - Grid search simplificado\n")
        global GRID_SEARCH_CONFIGS, EPOCHS
        for model_name in GRID_SEARCH_CONFIGS:
            GRID_SEARCH_CONFIGS[model_name] = {
                k: [v[0]] if isinstance(v, list) else [v]
                for k, v in GRID_SEARCH_CONFIGS[model_name].items()
            }
            GRID_SEARCH_CONFIGS[model_name]['epochs'] = [2]
            GRID_SEARCH_CONFIGS[model_name]['patience'] = [2]
    
    # MLflow
    mlflow_uri = args.mlflow_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Models
    models_to_run = args.models if args.models else list(GRID_SEARCH_CONFIGS.keys())
    
    # Datasets
    datasets_to_run = args.datasets if args.datasets else list(DATASETS.keys())
    
    # Backbone methods
    backbone_methods_to_run = args.backbone_methods if args.backbone_methods else list(BACKBONE_METHODS.keys())
    
    config = {
        "device": device,
        "output_dir": output_dir,
        "models": models_to_run,
        "datasets": datasets_to_run,
        "backbone_methods": backbone_methods_to_run,
        "skip_original": args.skip_original or args.backbone_only,
        "backbone_only": args.backbone_only,
        "quick_test": args.quick_test
    }

    return config


def run_original_experiments(config, experiments_list):
    """
    Executa experimentos com dados originais (sem backbone)
    
    Args:
        config: Configuração do experimento
        experiments_list: Lista para armazenar resultados
    """
    if config["skip_original"]:
        print("\n⏭️  Pulando experimentos com dados originais")
        return
    
    print("\n" + "="*80)
    print("EXECUTANDO EXPERIMENTOS COM DADOS ORIGINAIS (SEM BACKBONE)")
    print("="*80)
    
    for dataset_name in config["datasets"]:
        if dataset_name not in DATASETS:
            
            print(f"⚠️  Dataset '{dataset_name}' não encontrado, pulando...")
            continue
        
        dataset_config = DATASETS[dataset_name]
        
        # Preparar dataloaders
        data_file = H5_DIR / dataset_config["h5_file"]
        adj_file = PKL_DIR / dataset_config["pkl_file"]
        
        if not data_file.exists():
            print(f"⚠️  Arquivo de dados não encontrado: {data_file}, pulando...")
            continue
        
        if not adj_file.exists():
            print(f"⚠️  Arquivo de adjacência não encontrado: {adj_file}, pulando...")
            continue
        
        print(f"\n📊 Carregando dataset: {dataset_name} ({dataset_config['description']})")
        
        train_loader, val_loader, test_loader, num_nodes, adj_mx, norm_stats = prepare_dataloaders(
            data_file=data_file,
            adj_file=adj_file,
            dataset_key=dataset_config["key"],
            seq_len=SEQ_LEN,
            horizon=HORIZON,
            batch_size=BATCH_SIZE,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        
        # Executar para cada modelo
        for model_name in config["models"]:
            if model_name not in GRID_SEARCH_CONFIGS:
                print(f"⚠️  Modelo '{model_name}' não tem configuração de grid search, pulando...")
                continue
            
            experiment_name = f"{model_name}_GridSearch_{dataset_name}_original"
            
            print(f"\n{'='*60}")
            print(f"🚀 Iniciando: {experiment_name}")
            print(f"{'='*60}")
            
            # Obter função de grid search
            grid_search_fn = model_registry.get_grid_search_fn(model_name)
            param_grid = GRID_SEARCH_CONFIGS[model_name]
            
            # Executar grid search
            all_results, best_result, df_best = grid_search_fn(
                param_grid=param_grid,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                adj_mx=adj_mx,
                num_nodes=num_nodes,
                experiment_name=experiment_name,
                device=config["device"]
            )
            
            # Armazenar resultados
            experiments_list.append({
                "experiment_name": f"{model_name} - {dataset_name.upper()} (Original)",
                "model": model_name,
                "dataset": dataset_name.upper(),
                "df_best": df_best,
                "metadata": {
                    "num_nodes": num_nodes,
                    "backbone": "None",
                    "data_split": f"{TRAIN_RATIO}-{VAL_RATIO}-{TEST_RATIO}",
                    "seq_len": SEQ_LEN,
                    "horizon": HORIZON,
                    "normalization": norm_stats.get("method", "none")
                }
            })


def run_backbone_experiments(config, experiments_list):
    """
    Executa experimentos com backbones
    
    Args:
        config: Configuração do experimento
        experiments_list: Lista para armazenar resultados
    """
    print("\n" + "="*80)
    print("EXECUTANDO EXPERIMENTOS COM BACKBONES")
    print("="*80)
    
    for dataset_name in config["datasets"]:
        if dataset_name not in DATASETS:
            print(f"⚠️  Dataset '{dataset_name}' não encontrado, pulando...")
            continue
        
        dataset_config = DATASETS[dataset_name]
        
        for method in config["backbone_methods"]:
            if method not in BACKBONE_METHODS:
                print(f"⚠️  Método de backbone '{method}' não encontrado, pulando...")
                continue
            
            for cutoff in BACKBONE_METHODS[method]:
                backbone_name = f"{dataset_name}-{method}-{cutoff}"
                graphml_file = GRAPHML_DIR / f"{backbone_name}.GraphML"
                
                if not graphml_file.exists():
                    print(f"⚠️  Backbone não encontrado: {graphml_file}, pulando...")
                    continue
                
                print(f"\n📊 Carregando backbone: {backbone_name}")
                
                # Carregar backbone
                backbone_adj = load_graphml_backbone(graphml_file)
                
                # Preparar dataloaders com backbone
                data_file = H5_DIR / dataset_config["h5_file"]
                adj_file = PKL_DIR / dataset_config["pkl_file"]
                
                print(f"📁 Arquivo de adjacência: {adj_file}")
                print(f"📁 Arquivo de dados: {data_file}")
                
                # train_loader, val_loader, test_loader, num_nodes, _, norm_stats = prepare_dataloaders(
                #     data_file=data_file,
                #     adj_file=adj_file,
                #     dataset_key=dataset_config["key"],
                #     seq_len=SEQ_LEN,
                #     horizon=HORIZON,
                #     batch_size=BATCH_SIZE,
                #     train_ratio=TRAIN_RATIO,
                #     val_ratio=VAL_RATIO,
                #     test_ratio=TEST_RATIO,
                #     backbone_adj_mx=backbone_adj,
                #     num_workers=NUM_WORKERS,
                #     pin_memory=PIN_MEMORY
                # )
                
                # # Executar para cada modelo
                # for model_name in config["models"]:
                #     if model_name not in GRID_SEARCH_CONFIGS:
                #         print(f"⚠️  Modelo '{model_name}' não tem configuração, pulando...")
                #         continue
                    
                #     experiment_name = f"{model_name}_GridSearch_{backbone_name}"
                    
                #     print(f"\n{'='*60}")
                #     print(f"🚀 Iniciando: {experiment_name}")
                #     print(f"{'='*60}")
                    
                #     # Obter função de grid search
                #     grid_search_fn = model_registry.get_grid_search_fn(model_name)
                #     param_grid = GRID_SEARCH_CONFIGS[model_name]
                    
                #     # Executar grid search
                #     all_results, best_result, df_best = grid_search_fn(
                #         param_grid=param_grid,
                #         train_loader=train_loader,
                #         val_loader=val_loader,
                #         test_loader=test_loader,
                #         adj_mx=backbone_adj,
                #         num_nodes=num_nodes,
                #         experiment_name=experiment_name,
                #         device=config["device"]
                #     )
                    
                #     # Armazenar resultados
                #     experiments_list.append({
                #         "experiment_name": f"{model_name} - {backbone_name}",
                #         "model": model_name,
                #         "dataset": f"{dataset_name.upper()}-{method.upper()}-{cutoff}",
                #         "df_best": df_best,
                #         "metadata": {
                #             "num_nodes": num_nodes,
                #             "backbone": f"{method}-{cutoff}",
                #             "data_split": f"{TRAIN_RATIO}-{VAL_RATIO}-{TEST_RATIO}",
                #             "seq_len": SEQ_LEN,
                #             "horizon": HORIZON,
                #             "normalization": norm_stats.get("method", "none")
                #         }
                #     })


def main():
    """Função principal"""
    
    print("\n" + "="*80)
    print("GNN TRAFFIC FORECASTING - PIPELINE DE EXPERIMENTOS")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse argumentos
    args = parse_arguments()
    
    # Setup
    config = setup_experiment(args)
    
    print(f"\n📋 Configuração:")
    print(f"   - Device: {config['device']}")
    print(f"   - Modelos: {config['models']}")
    print(f"   - Datasets: {config['datasets']}")
    print(f"   - Métodos de backbone: {config['backbone_methods']}")
    print(f"   - Diretório de saída: {config['output_dir']}")
    print(f"   - Quick test: {config['quick_test']}")
    
    # Registrar modelos
    print("\n🔧 Registrando modelos...")
    register_default_models()
    
    # Validar modelos solicitados
    for model_name in config["models"]:
        if not model_registry.is_registered(model_name):
            print(f"❌ Erro: Modelo '{model_name}' não está registrado!")
            print(f"   Modelos disponíveis: {model_registry.list_models()}")
            sys.exit(1)
    
    # Lista para armazenar todos os experimentos
    experiments_list = []
    
    try:
        # Executar experimentos originais
        if not config["backbone_only"]:
            run_original_experiments(config, experiments_list)
        
        # Executar experimentos com backbone
        if not config["skip_original"]:
            run_backbone_experiments(config, experiments_list)
        
        # Consolidar resultados
        print("\n" + "="*80)
        print("CONSOLIDANDO RESULTADOS")
        print("="*80)
        
        if not experiments_list:
            print("❌ Nenhum experimento foi executado!")
            return
        
        df_consolidated = consolidate_experiment_results(
            experiments_data=experiments_list,
            output_csv="all_experiments_consolidated.csv",
            output_json="all_experiments_detailed.json",
            primary_metric=PRIMARY_METRIC,
            save_path=config["output_dir"]
        )
        
        # Criar relatório comparativo
        if not df_consolidated.empty:
            create_comparison_report(
                consolidated_df=df_consolidated,
                output_file="comparison_report.md",
                save_path=config["output_dir"]
            )
            
            export_best_configs_to_json(
                consolidated_df=df_consolidated,
                output_file="best_configs.json",
                save_path=config["output_dir"]
            )
        
        print("\n" + "="*80)
        print("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print(f"   Total de experimentos: {len(experiments_list)}")
        print(f"   Resultados salvos em: {config['output_dir']}")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida pelo usuário")
        print(f"   Experimentos concluídos: {len(experiments_list)}")
        
        if experiments_list:
            print("\n💾 Salvando resultados parciais...")
            df_consolidated = consolidate_experiment_results(
                experiments_data=experiments_list,
                output_csv="partial_experiments_consolidated.csv",
                output_json="partial_experiments_detailed.json",
                primary_metric=PRIMARY_METRIC,
                save_path=config["output_dir"]
            )
        
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()