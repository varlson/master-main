# Relatório de Comparação de Experimentos GNN

**Gerado em:** 2026-03-06 17:30:47

**Total de experimentos:** 6

## Resumo por Modelo

### DCRNN

- Experimentos: 1
- MAE médio: 0.2860 ± nan
- RMSE médio: 0.5349 ± nan
- MAPE médio: 169.69% ± nan%

### GraphWaveNet

- Experimentos: 1
- MAE médio: 0.2152 ± nan
- RMSE médio: 0.4528 ± nan
- MAPE médio: 121.46% ± nan%

### MTGNN

- Experimentos: 1
- MAE médio: 0.2405 ± nan
- RMSE médio: 0.5078 ± nan
- MAPE médio: 135.36% ± nan%

### DGCRN

- Experimentos: 1
- MAE médio: 0.2443 ± nan
- RMSE médio: 0.5003 ± nan
- MAPE médio: 164.52% ± nan%

### STICformer

- Experimentos: 1
- MAE médio: 0.2036 ± nan
- RMSE médio: 0.4333 ± nan
- MAPE médio: 123.63% ± nan%

### PatchSTG

- Experimentos: 1
- MAE médio: 0.2075 ± nan
- RMSE médio: 0.4350 ± nan
- MAPE médio: 123.13% ± nan%

## Melhores Resultados por Dataset

### pems-bay

- **Melhor modelo:** STICformer
- **Experimento:** pems-bay_STICformer_20260306_142338
- **MAE:** 0.2036
- **RMSE:** 0.4333
- **MAPE:** 123.63%
- **Parâmetros:** `{'input_dim': 1, 'hidden_dim': 64, 'output_dim': 1, 'seq_len': 12, 'horizon': 12, 'num_layers': 2, 'num_heads': 4, 'ff_multiplier': 2, 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 200, 'patience': 5}`

## Tabela Completa de Resultados

| Experimento | Modelo | Dataset | MAE | RMSE | MAPE (%) |
|-------------|--------|---------|-----|------|----------|
| pems-bay_DCRNN_20260306_142338 | DCRNN | pems-bay | 0.2860 | 0.5349 | 169.69 |
| pems-bay_GraphWaveNet_20260306_142338 | GraphWaveNet | pems-bay | 0.2152 | 0.4528 | 121.46 |
| pems-bay_MTGNN_20260306_142338 | MTGNN | pems-bay | 0.2405 | 0.5078 | 135.36 |
| pems-bay_DGCRN_20260306_142338 | DGCRN | pems-bay | 0.2443 | 0.5003 | 164.52 |
| pems-bay_STICformer_20260306_142338 | STICformer | pems-bay | 0.2036 | 0.4333 | 123.63 |
| pems-bay_PatchSTG_20260306_142338 | PatchSTG | pems-bay | 0.2075 | 0.4350 | 123.13 |
