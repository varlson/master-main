# GNN Traffic Forecasting Pipeline

Pipeline modular e extensível para experimentos com Graph Neural Networks (GNNs) em previsão de tráfego usando datasets como METR-LA e PEMS-BAY.

## 📁 Estrutura do Projeto

```
.
├── main.py                      # Script principal
├── config.py                    # Configurações centralizadas
├── grid_search.py              # Funções de grid search para modelos
├── models/
│   ├── DCRNN.py                # Modelo DCRNN
│   └── WaveNet.py              # Modelo GraphWaveNet
├── utils/
│   ├── data_loader.py          # Carregamento e preprocessamento de dados
│   ├── consolidate_results.py # Consolidação de resultados
│   └── model_registry.py       # Registry de modelos
├── dataset/
│   ├── h5/                     # Arquivos .h5 com dados
│   └── pkl/                    # Arquivos .pkl com matrizes de adjacência
├── output/
│   └── graphml/                # Backbones em formato GraphML
├── results/                    # Resultados dos experimentos
└── mlruns/                     # MLflow tracking

```

## 🚀 Uso Rápido

### Instalação de Dependências

```bash
pip install torch numpy scipy pandas h5py networkx mlflow scikit-learn
```

### Execução Básica

```bash
# Executar todos os modelos em todos os datasets
python main.py

# Executar modelos específicos
python main.py --models DCRNN GraphWaveNet

# Executar em datasets específicos
python main.py --datasets metr-la pems-bay

# Apenas experimentos com backbone
python main.py --backbone-only

# Modo de teste rápido (para debug)
python main.py --quick-test

# Especificar device
python main.py --device cuda:0
```

### Opções Disponíveis

```bash
python main.py --help
```

- `--models`: Lista de modelos a executar
- `--datasets`: Lista de datasets a usar
- `--skip-original`: Pular experimentos sem backbone
- `--backbone-only`: Apenas experimentos com backbone
- `--backbone-methods`: Métodos de backbone específicos
- `--quick-test`: Modo de teste rápido
- `--device`: Device PyTorch (cpu, cuda, cuda:0, etc.)
- `--output-dir`: Diretório para salvar resultados
- `--mlflow-uri`: URI do MLflow tracking

## 📊 Datasets Suportados

### METR-LA

- **Descrição**: Tráfego da rede Metro de Los Angeles
- **Nós**: 207 sensores
- **Formato**: H5 + PKL (matriz de adjacência)

### PEMS-BAY

- **Descrição**: Tráfego da Bay Area
- **Nós**: 325 sensores
- **Formato**: H5 + PKL (matriz de adjacência)

### Adicionar Novo Dataset

Edite `config.py`:

```python
DATASETS = {
    # ... datasets existentes

    "novo-dataset": {
        "h5_file": "NOVO-DATASET.h5",
        "pkl_file": "adj_mx_novo.pkl",
        "key": "data",  # chave no arquivo H5
        "description": "Descrição do dataset",
        "num_nodes": None,  # detectado automaticamente
        "feature": "speed"
    }
}
```

Coloque os arquivos em:

- `dataset/h5/NOVO-DATASET.h5`
- `dataset/pkl/adj_mx_novo.pkl`

## 🧠 Modelos Disponíveis

### 1. DCRNN (Diffusion Convolutional Recurrent Neural Network)

- **Arquitetura**: Encoder-Decoder com células GRU e convolução em grafos
- **Hiperparâmetros principais**:
  - `hidden_dim`: Dimensão oculta
  - `k`: Ordem da difusão
  - `dropout`: Taxa de dropout
  - `use_scheduled_sampling`: Scheduled sampling durante treinamento

### 2. GraphWaveNet

- **Arquitetura**: Convoluções temporais dilatadas + convoluções em grafos
- **Hiperparâmetros principais**:
  - `hidden_dim`: Dimensão oculta
  - `num_blocks`: Número de blocos WaveNet
  - `dilation_base`: Base para dilatação
  - `k`: Ordem da convolução no grafo

## ➕ Adicionar Novo Modelo

### Passo 1: Criar arquivo do modelo

Crie `models/NovoModelo.py`:

```python
import torch
import torch.nn as nn

class NovoModelo(nn.Module):
    def __init__(
        self,
        adj_mx,
        num_nodes,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        seq_len=12,
        horizon=12,
        lr=1e-3,
        weight_decay=0.0,
        epochs=50,
        patience=10,
        device='cpu'
    ):
        super().__init__()
        # Inicialização do modelo
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.horizon = horizon
        self.device = device
        self.epochs = epochs
        self.patience = patience

        # Camadas do modelo
        # ...

        # Otimizador
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.loss_fn = nn.MSELoss()
        self.to(device)

    def forward(self, x):
        # Forward pass
        pass

    def fit(self, train_loader, val_loader=None):
        # Treinamento
        pass

    def evaluate(self, loader):
        # Avaliação
        pass

    def predict(self, loader):
        # Predição
        pass
```

### Passo 2: Criar função de grid search

Em `grid_search.py`, adicione:

```python
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
    # Implementação similar a DCRNN_grid_search
    # Ver grid_search.py para referência
    pass
```

### Passo 3: Registrar o modelo

Em `main.py`, adicione após `register_default_models()`:

```python
from models.NovoModelo import NovoModelo
from grid_search import NovoModelo_grid_search

model_registry.register_model(
    name="NovoModelo",
    model_class=NovoModelo,
    grid_search_fn=NovoModelo_grid_search,
    default_params={
        "input_dim": 1,
        "output_dim": 1,
        "hidden_dim": 64,
        # ... outros parâmetros padrão
    }
)
```

### Passo 4: Adicionar configuração de grid search

Em `config.py`:

```python
GRID_SEARCH_CONFIGS["NovoModelo"] = {
    "input_dim": [1],
    "output_dim": [1],
    "seq_len": [SEQ_LEN],
    "horizon": [HORIZON],
    "hidden_dim": [32, 64, 128],
    # ... outros hiperparâmetros para busca
    "epochs": [EPOCHS],
    "patience": [EARLY_STOPPING_PATIENCE]
}
```

### Passo 5: Executar

```bash
python main.py --models NovoModelo --datasets metr-la
```

## 🔬 Experimentos com Backbone

O pipeline suporta experimentos com backbones de rede extraídos por diferentes métodos:

### Métodos Disponíveis

1. **Disparity**: Filtragem por disparidade
2. **Noise**: Correção por ruído
3. **Threshold**: Limiarização

### Configuração

Em `config.py`:

```python
BACKBONE_METHODS = {
    "disparity": [0.1, 0.2, 0.3, 0.4, 0.5],
    "noise": [0.1, 0.2, 0.3, 0.4, 0.5],
    "threshold": [0.1, 0.2, 0.3, 0.4, 0.5]
}
```

### Formato dos Backbones

Os backbones devem estar em formato GraphML em:

```
output/graphml/{dataset}-{method}-{cutoff}.GraphML
```

Exemplo: `output/graphml/metr-la-disparity-0.3.GraphML`

## 📈 MLflow Tracking

Todos os experimentos são rastreados automaticamente com MLflow:

```bash
# Visualizar experimentos
mlflow ui

# Acessar em: http://localhost:5000
```

### Métricas Registradas

- `train_loss`: Loss de treino por época
- `val_loss`: Loss de validação por época
- `test_loss`: Loss final no conjunto de teste
- `test_mae`: Mean Absolute Error
- `test_rmse`: Root Mean Square Error
- `test_mape`: Mean Absolute Percentage Error
- `learning_rate`: Taxa de aprendizado por época
- `best_val_loss`: Melhor loss de validação
- `early_stop_epoch`: Época de early stopping (se aplicável)

## 📊 Resultados

Os resultados são salvos em múltiplos formatos:

### 1. CSV Consolidado

`results/all_experiments_consolidated.csv`

Contém uma linha por experimento com métricas principais.

### 2. JSON Detalhado

`results/all_experiments_detailed.json`

Contém informações completas de todos os experimentos.

### 3. Relatório Markdown

`results/comparison_report.md`

Relatório comparativo formatado com:

- Resumo por modelo
- Melhores resultados por dataset
- Tabela completa de resultados

### 4. Melhores Configurações

`results/best_configs.json`

Melhores configurações de hiperparâmetros por dataset e modelo.

## ⚙️ Configurações Avançadas

### Ajustar Hiperparâmetros de Treino

Em `config.py`:

```python
# Divisão dos dados
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# Sequências temporais
SEQ_LEN = 12  # Janela de entrada
HORIZON = 12  # Horizonte de previsão

# Batch size
BATCH_SIZE = 64

# Treinamento
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
```

### Modo Debug

Para testes rápidos, ative o modo debug em `config.py`:

```python
DEBUG_MODE = True
```

Ou use a flag na linha de comando:

```bash
python main.py --quick-test
```

## 🐛 Troubleshooting

### Erro: "Arquivo não encontrado"

- Verifique se os arquivos `.h5` e `.pkl` estão nos diretórios corretos
- Verifique se os backbones `.GraphML` existem (se usando `--backbone-only`)

### Erro: "CUDA out of memory"

- Reduza `BATCH_SIZE` em `config.py`
- Use CPU: `python main.py --device cpu`

### Erro: "Modelo não registrado"

- Verifique se chamou `register_default_models()`
- Verifique se adicionou o modelo no registry

### Resultados parciais após interrupção

- Os resultados parciais são salvos automaticamente em caso de `Ctrl+C`
- Procure por `partial_experiments_consolidated.csv`

## 📚 Referências

### DCRNN

- Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. ICLR 2018.

### GraphWaveNet

- Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling. IJCAI 2019.

## 📝 Licença

Este projeto é fornecido como está para fins acadêmicos e de pesquisa.

## 👥 Contribuindo

Para adicionar novos modelos, datasets ou funcionalidades, siga os guias acima e mantenha a estrutura modular do código.
