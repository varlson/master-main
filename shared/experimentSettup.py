from config import DEVICE, MLFLOW_TRACKING_URI, DATASETS, BACKBONE_METHODS, GRID_SEARCH_CONFIGS
import mlflow
from pathlib import Path

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
    
    print(backbone_methods_to_run)
    
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