# Relatório de Comparação de Experimentos GNN

**Gerado em:** 2026-03-04 01:50:11

**Total de experimentos:** 6

## Resumo por Modelo

### DCRNN

- Experimentos: 1
- MAE médio: 0.3222 ± nan
- RMSE médio: 0.6241 ± nan
- MAPE médio: 124.45% ± nan%

### GraphWaveNet

- Experimentos: 1
- MAE médio: 0.3133 ± nan
- RMSE médio: 0.6192 ± nan
- MAPE médio: 111.88% ± nan%

### MTGNN

- Experimentos: 1
- MAE médio: 0.3433 ± nan
- RMSE médio: 0.6347 ± nan
- MAPE médio: 123.83% ± nan%

### DGCRN

- Experimentos: 1
- MAE médio: 0.2892 ± nan
- RMSE médio: 0.6294 ± nan
- MAPE médio: 123.01% ± nan%

### STICformer

- Experimentos: 1
- MAE médio: 0.2918 ± nan
- RMSE médio: 0.6128 ± nan
- MAPE médio: 111.02% ± nan%

### PatchSTG

- Experimentos: 1
- MAE médio: 0.3085 ± nan
- RMSE médio: 0.6138 ± nan
- MAPE médio: 115.22% ± nan%

## Melhores Resultados por Dataset

### metr-la

- **Melhor modelo:** DGCRN
- **Experimento:** metr-la_DGCRN_20260303_202038
- **MAE:** 0.2892
- **RMSE:** 0.6294
- **MAPE:** 123.01%
- **Parâmetros:** `{'input_dim': 1, 'hidden_dim': 64, 'output_dim': 1, 'seq_len': 12, 'horizon': 12, 'node_dim': 16, 'gcn_depth': 2, 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 30, 'patience': 5}`

## Tabela Completa de Resultados

| Experimento | Modelo | Dataset | MAE | RMSE | MAPE (%) |
|-------------|--------|---------|-----|------|----------|
| metr-la_DCRNN_20260303_202038 | DCRNN | metr-la | 0.3222 | 0.6241 | 124.45 |
| metr-la_GraphWaveNet_20260303_202038 | GraphWaveNet | metr-la | 0.3133 | 0.6192 | 111.88 |
| metr-la_MTGNN_20260303_202038 | MTGNN | metr-la | 0.3433 | 0.6347 | 123.83 |
| metr-la_DGCRN_20260303_202038 | DGCRN | metr-la | 0.2892 | 0.6294 | 123.01 |
| metr-la_STICformer_20260303_202038 | STICformer | metr-la | 0.2918 | 0.6128 | 111.02 |
| metr-la_PatchSTG_20260303_202038 | PatchSTG | metr-la | 0.3085 | 0.6138 | 115.22 |
