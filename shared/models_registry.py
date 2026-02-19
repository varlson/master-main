"""
Registry de modelos para facilitar adição de novos modelos
"""

from typing import Dict, Callable, Any
from pathlib import Path
import importlib
import sys


class ModelRegistry:
    """
    Registry centralizado para gerenciar modelos GNN
    """
    
    def __init__(self):
        self._models = {}
        self._grid_search_functions = {}
        
    def register_model(
        self,
        name: str,
        model_class: Any,
        grid_search_fn: Callable,
        default_params: Dict = None
    ):
        """
        Registra um novo modelo
        
        Args:
            name: Nome do modelo
            model_class: Classe do modelo
            grid_search_fn: Função de grid search
            default_params: Parâmetros padrão
        """
        self._models[name] = {
            "class": model_class,
            "grid_search": grid_search_fn,
            "default_params": default_params or {}
        }
        print(f"✓ Modelo '{name}' registrado")
    
    def get_model_class(self, name: str):
        """Retorna a classe do modelo"""
        if name not in self._models:
            raise ValueError(f"Modelo '{name}' não registrado. "
                           f"Modelos disponíveis: {list(self._models.keys())}")
        return self._models[name]["class"]
    
    def get_grid_search_fn(self, name: str):
        """Retorna a função de grid search"""
        if name not in self._models:
            raise ValueError(f"Modelo '{name}' não registrado")
        return self._models[name]["grid_search"]
    
    def get_default_params(self, name: str):
        """Retorna parâmetros padrão"""
        if name not in self._models:
            raise ValueError(f"Modelo '{name}' não registrado")
        return self._models[name]["default_params"].copy()
    
    def list_models(self):
        """Lista todos os modelos registrados"""
        return list(self._models.keys())
    
    def is_registered(self, name: str) -> bool:
        """Verifica se um modelo está registrado"""
        return name in self._models


# Instância global do registry
model_registry = ModelRegistry()


def register_default_models():
    """
    Registra os modelos padrão (DCRNN e GraphWaveNet)
    """
    try:
        # Importar modelos
        from models.DCRNN import DCRNN
        from models.WaveNet import GraphWaveNet
        from shared  import DCRNN_grid_search, GraphWaveNet_grid_search
        
        # Registrar DCRNN
        model_registry.register_model(
            name="DCRNN",
            model_class=DCRNN,
            grid_search_fn=DCRNN_grid_search,
            default_params={
                "input_dim": 1,
                "output_dim": 1,
                "hidden_dim": 64,
                "k": 2,
                "dropout": 0.1,
                "lr": 1e-3,
                "weight_decay": 1e-3,
                "use_scheduled_sampling": False,
                "teacher_forcing_ratio": 0.5
            }
        )
        
        # Registrar GraphWaveNet
        model_registry.register_model(
            name="GraphWaveNet",
            model_class=GraphWaveNet,
            grid_search_fn=GraphWaveNet_grid_search,
            default_params={
                "input_dim": 1,
                "output_dim": 1,
                "hidden_dim": 64,
                "num_blocks": 4,
                "dilation_base": 2,
                "k": 2,
                "dropout": 0.1,
                "lr": 1e-3,
                "weight_decay": 1e-4
            }
        )
        
        print(f"✓ Modelos padrão registrados: {model_registry.list_models()}")
        
    except ImportError as e:
        print(f"⚠️  Erro ao importar modelos padrão: {e}")
        print("   Certifique-se de que os módulos models.DCRNN e models.WaveNet existem")


def load_custom_model(
    model_path: Path,
    model_name: str,
    class_name: str,
    grid_search_fn_name: str
):
    """
    Carrega um modelo personalizado de um arquivo Python
    
    Args:
        model_path: Caminho para o arquivo .py do modelo
        model_name: Nome para registrar o modelo
        class_name: Nome da classe do modelo
        grid_search_fn_name: Nome da função de grid search
    
    Example:
        load_custom_model(
            model_path=Path("models/MyModel.py"),
            model_name="MyModel",
            class_name="MyGNNModel",
            grid_search_fn_name="MyModel_grid_search"
        )
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")
    
    # Adicionar diretório ao path
    sys.path.insert(0, str(model_path.parent))
    
    # Importar módulo
    module_name = model_path.stem
    module = importlib.import_module(module_name)
    
    # Obter classe e função
    try:
        model_class = getattr(module, class_name)
        grid_search_fn = getattr(module, grid_search_fn_name)
        
        # Registrar
        model_registry.register_model(
            name=model_name,
            model_class=model_class,
            grid_search_fn=grid_search_fn
        )
        
        print(f"✓ Modelo personalizado '{model_name}' carregado e registrado")
        
    except AttributeError as e:
        raise AttributeError(
            f"Não foi possível encontrar '{class_name}' ou '{grid_search_fn_name}' "
            f"em {model_path}: {e}"
        )


# Template para adicionar novo modelo
TEMPLATE_NEW_MODEL = """
# ============================================
# TEMPLATE PARA NOVO MODELO
# ============================================

# 1. Criar arquivo models/NovoModelo.py com a classe do modelo

# 2. Criar função de grid search em grid_search.py:

def NovoModelo_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="NovoModelo_GridSearch",
    device='cpu'
):
    # Implementação similar a DCRNN_grid_search ou GraphWaveNet_grid_search
    pass

# 3. Registrar no main.py:

from models.NovoModelo import NovoModelo
from grid_search import NovoModelo_grid_search

model_registry.register_model(
    name="NovoModelo",
    model_class=NovoModelo,
    grid_search_fn=NovoModelo_grid_search,
    default_params={...}
)

# 4. Adicionar configuração em config.py:

GRID_SEARCH_CONFIGS["NovoModelo"] = {
    "input_dim": [1],
    "output_dim": [1],
    # ... outros hiperparâmetros
}

# Pronto! O modelo estará disponível no pipeline
"""