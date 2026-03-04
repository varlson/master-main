# Forecasting Espaço-Temporal em Redes de Tráfego (GNN)

Projeto para experimentação e comparação de modelos de previsão de tráfego em séries temporais com estrutura de grafo, usando datasets como `METR-LA` e `PEMS-BAY`.

O pipeline atual suporta:
- preparação de dados a partir de arquivos `.npy`
- treino e grid search com MLflow
- execução em múltiplos datasets na mesma rodada
- consolidação de resultados em CSV/JSON/Markdown

## Objetivo

Comparar modelos espaço-temporais (GNN / RNN / conv-temporal) em datasets de tráfego, mantendo um fluxo único de:

1. carga dos dados
2. criação de `DataLoaders`
3. treino/validação/teste
4. grid search
5. consolidação e relatório dos resultados

## Modelos Implementados

- `DCRNN` (`models/DCRNN.py`)
- `GraphWaveNet` (`models/WaveNet.py`)
- `MTGNN` (`models/MTGNN.py`)
- `DGCRN` (`models/DGCRN.py`)
- `STICformer` (`models/STICformer.py`)
- `PatchSTG` (`models/PatchSTG.py`)

Todos os modelos foram integrados com:
- `fit(train_loader, val_loader)`
- `evaluate(loader)`
- `predict(loader)`

## Datasets Suportados (estado atual)

Atualmente o projeto já está preparado para:
- `metr-la`
- `pems-bay`

Os arquivos devem estar em `data/npy` com o padrão:
- `data/npy/<dataset>-h5.npy` (série temporal)
- `data/npy/<dataset>-adj_mx.npy` (matriz de adjacência)

Exemplos:
- `data/npy/metr-la-h5.npy`
- `data/npy/metr-la-adj_mx.npy`
- `data/npy/pems-bay-h5.npy`
- `data/npy/pems-bay-adj_mx.npy`

## Estrutura do Projeto

```text
.
├── main.py                       # Pipeline principal (multi-dataset + consolidação)
├── requirements.txt              # Dependências (snapshot amplo do ambiente)
├── models/
│   ├── DCRNN.py                  # Modelo DCRNN
│   ├── WaveNet.py                # Modelo GraphWaveNet
│   ├── MTGNN.py                  # Modelo MTGNN
│   ├── DGCRN.py                  # Modelo DGCRN
│   ├── STICformer.py             # Modelo STICformer
│   └── PatchSTG.py               # Modelo PatchSTG
├── shared/
│   ├── loaders.py                # Preparo de dados, normalização e DataLoaders
│   ├── MLFlow.py                 # Treino + grid search + integração com MLflow
│   ├── resultSumarization.py     # Consolidação CSV/JSON/relatório
│   ├── dataprocessor.py          # Conversão de arquivos (h5/pkl -> npy)
│   └── utils.py                  # Utilitários diversos (grafo, download, etc.)
├── data/
│   ├── h5/                       # Dados originais .h5 (opcional para conversão)
│   ├── pkl/                      # Matrizes/objetos .pkl (opcional para conversão)
│   └── npy/                      # Entrada principal do pipeline
├── backbone/
│   ├── disparity_filter.py       # Extração de backbone (Disparity Filter)
│   └── noise_corrected.py        # Extração de backbone (Noise Corrected)
└── results/                      # Saídas consolidadas e sumários de grid search
```

## Requisitos e Instalação

## Requisitos

- Python 3.10+ (recomendado)
- `pip`
- (Opcional) GPU com CUDA para acelerar treino

## Instalação

Crie um ambiente virtual e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Observação:
- `requirements.txt` contém um conjunto grande de pacotes (incluindo CUDA e libs extras). Em alguns ambientes pode ser necessário ajustar versões.

## Como Usar

O pipeline é executado por `main.py` e usa variáveis de ambiente para configuração.

## Execução básica (dataset único)

```bash
python3 main.py
```

Por padrão:
- dataset: `pems-bay`
- modelos: `DCRNN`, `GraphWaveNet`, `MTGNN`, `DGCRN`, `STICformer`, `PatchSTG`
- `seq_len=12`, `horizon=12`, `batch_size=8`

## Executar dataset específico

```bash
DATASET_NAME=metr-la python3 main.py
```

## Executar múltiplos datasets na mesma rodada

`DATASET_NAMES` tem prioridade sobre `DATASET_NAME`.

```bash
DATASET_NAMES="pems-bay,metr-la" python3 main.py
```

Isso gera:
- consolidação por dataset (`pems-bay_*`, `metr-la_*`)
- consolidação global (`all-datasets_*`)

## Ativar/desativar modelos

```bash
RUN_DCRNN=1 RUN_GRAPH_WAVENET=0 RUN_MTGNN=1 RUN_DGCRN=1 RUN_STICFORMER=1 RUN_PATCHSTG=1 python3 main.py
```

Valores aceitos na prática:
- `1` para habilitar
- `0` para desabilitar

## Forçar dispositivo

```bash
DEVICE=cpu python3 main.py
```

ou

```bash
DEVICE=cuda python3 main.py
```

## Ajustar janela/horizonte/batch

```bash
SEQ_LEN=12 HORIZON=12 BATCH_SIZE=32 python3 main.py
```

## Variáveis de Ambiente (Pipeline)

- `DATASET_NAME`: dataset único (fallback)
- `DATASET_NAMES`: lista separada por vírgula (`"pems-bay,metr-la"`)
- `DEVICE`: `cpu` ou `cuda`
- `SEQ_LEN`: tamanho da janela de entrada
- `HORIZON`: horizonte de previsão
- `BATCH_SIZE`: tamanho do batch
- `RUN_DCRNN`: `1`/`0`
- `RUN_GRAPH_WAVENET`: `1`/`0`
- `RUN_MTGNN`: `1`/`0`
- `RUN_DGCRN`: `1`/`0`
- `RUN_STICFORMER`: `1`/`0`
- `RUN_PATCHSTG`: `1`/`0`
- `GENERATE_PLOTS`: `1`/`0` (gera diagnosticos visuais por experimento)
- `PLOTS_NUM_NODES`: quantidade de nos no grafico "Real vs Previsto"
- `PLOTS_MAX_TIME_POINTS`: limite de pontos no eixo temporal (linhas/heatmap)

## Fluxo do Pipeline

Para cada dataset:

1. valida presença dos arquivos `.npy`
2. carrega série temporal e matriz de adjacência
3. prepara `train/val/test` com normalização e janelas temporais
4. executa grid search dos modelos habilitados
5. consolida resultados do dataset em `results/`

Se houver mais de um dataset na execução:

6. gera consolidação global `all-datasets_<RUN_ID>_*`

## Saídas Geradas

### Sumários de Grid Search (por modelo/experimento)

Gerados por `shared/MLFlow.py` em:
- `results/<experiment_name>_summary.json`

### Consolidação (por dataset e global)

Gerados por `shared/resultSumarization.py`:

- `results/<scope>_<RUN_ID>_consolidated_experiments.csv`
- `results/<scope>_<RUN_ID>_consolidated_experiments.json`
- `results/<scope>_<RUN_ID>_comparison_report.md`
- `results/<scope>_<RUN_ID>_best_configs.json`

Onde `<scope>` pode ser:
- `pems-bay`
- `metr-la`
- `all-datasets`

### Diagnosticos Visuais (por experimento)

Gerados automaticamente por `shared/visualization.py` quando `GENERATE_PLOTS=1`:

- `results/plots/<experiment_name>/overall_metrics.json`
- `results/plots/<experiment_name>/metrics_by_horizon.csv`
- `results/plots/<experiment_name>/metrics_by_node.csv`
- `results/plots/<experiment_name>/real_vs_pred_nodes.png`
- `results/plots/<experiment_name>/metrics_by_horizon.png`
- `results/plots/<experiment_name>/metrics_by_node.png`
- `results/plots/<experiment_name>/error_over_time.png`
- `results/plots/<experiment_name>/train_val_curves.png` (quando houver historico de treino)
- `results/plots/<experiment_name>/error_heatmap_node_time.png`

## MLflow

O treino/grid search usa MLflow para:
- registrar hiperparâmetros
- registrar métricas (`train`, `val`, `test`)
- salvar artefatos de modelo (`mlflow.pytorch.log_model`)

Diretório local padrão (se não configurado):
- `mlruns/`

Para visualizar a UI local:

```bash
mlflow ui
```

## Conversão de Dados (opcional)

O projeto inclui utilitários para converter dados para `.npy`:

- `shared/dataprocessor.py`
  - `h5tonpy(...)`
  - `pkltonpy(...)`

Esses utilitários são úteis quando você recebe os dados em `.h5`/`.pkl` e quer preparar a entrada usada pelo pipeline principal.

## Extensão do Projeto (Adicionar Novos Modelos)

O padrão atual para integrar um novo modelo no pipeline é:

1. criar arquivo em `models/`
2. implementar API compatível:
   - `forward`
   - `fit`
   - `evaluate`
   - `predict`
3. adicionar `*_train_with_mlflow` e `*_grid_search` em `shared/MLFlow.py`
4. incluir o modelo no `main.py`:
   - flag `RUN_*`
   - grid de hiperparâmetros
   - chamada no loop por dataset

## Observações / Limitações Atuais

- Os grids em `main.py` estão enxutos por padrão (epocas baixas) para facilitar testes rápidos.
- Os modelos `MTGNN`, `DGCRN`, `STICformer` e `PatchSTG` foram integrados ao pipeline no mesmo padrão de interface dos demais.
- `requirements.txt` pode exigir ajuste dependendo do ambiente (CPU/GPU, CUDA, versões locais).

## Próximos Passos Recomendados

- expandir grids de hiperparâmetros por modelo
- adicionar métricas por horizonte (ex.: 3/6/12 passos)
- incluir novos modelos (ex.: AGCRN, STAEformer, PDFormer)
- documentar protocolo de benchmark (split, normalização, seeds)
