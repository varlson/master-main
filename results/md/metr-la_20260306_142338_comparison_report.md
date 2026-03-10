# Relatório de Comparação de Experimentos GNN

**Gerado em:** 2026-03-06 19:12:47

**Total de experimentos:** 6

## Resumo por Modelo

### DCRNN

- Experimentos: 1
- MAE médio: 0.3411 ± nan
- RMSE médio: 0.6264 ± nan
- MAPE médio: 123.25% ± nan%

### GraphWaveNet

- Experimentos: 1
- MAE médio: 0.3256 ± nan
- RMSE médio: 0.6207 ± nan
- MAPE médio: 113.15% ± nan%

### MTGNN

- Experimentos: 1
- MAE médio: 0.3529 ± nan
- RMSE médio: 0.6354 ± nan
- MAPE médio: 124.08% ± nan%

### DGCRN

- Experimentos: 1
- MAE médio: 0.3077 ± nan
- RMSE médio: 0.6284 ± nan
- MAPE médio: 124.19% ± nan%

### STICformer

- Experimentos: 1
- MAE médio: 0.3096 ± nan
- RMSE médio: 0.6089 ± nan
- MAPE médio: 113.71% ± nan%

### PatchSTG

- Experimentos: 1
- MAE médio: 0.3082 ± nan
- RMSE médio: 0.6086 ± nan
- MAPE médio: 111.69% ± nan%

## Melhores Resultados por Dataset

### metr-la

- **Melhor modelo:** DGCRN
- **Experimento:** metr-la_DGCRN_20260306_142338
- **MAE:** 0.3077
- **RMSE:** 0.6284
- **MAPE:** 124.19%
- **Parâmetros:** `{'input_dim': 1, 'hidden_dim': 64, 'output_dim': 1, 'seq_len': 12, 'horizon': 12, 'node_dim': 16, 'gcn_depth': 2, 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 200, 'patience': 5}`

## Tabela Completa de Resultados

| Experimento | Modelo | Dataset | MAE | RMSE | MAPE (%) |
|-------------|--------|---------|-----|------|----------|
| metr-la_DCRNN_20260306_142338 | DCRNN | metr-la | 0.3411 | 0.6264 | 123.25 |
| metr-la_GraphWaveNet_20260306_142338 | GraphWaveNet | metr-la | 0.3256 | 0.6207 | 113.15 |
| metr-la_MTGNN_20260306_142338 | MTGNN | metr-la | 0.3529 | 0.6354 | 124.08 |
| metr-la_DGCRN_20260306_142338 | DGCRN | metr-la | 0.3077 | 0.6284 | 124.19 |
| metr-la_STICformer_20260306_142338 | STICformer | metr-la | 0.3096 | 0.6089 | 113.71 |
| metr-la_PatchSTG_20260306_142338 | PatchSTG | metr-la | 0.3082 | 0.6086 | 111.69 |
