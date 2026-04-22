# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 09:56:51

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### pems-bay

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 2.1477 ± 0.0000 | 4.4732 ± 0.0000 | 3.44 ± 0.00 | 0.00 |
| 2 | MTGNN | 2.2812 ± 0.0000 | 4.8506 ± 0.0000 | 3.65 ± 0.00 | 6.22 |
| 3 | GraphWaveNet | 2.5944 ± 0.0000 | 4.9512 ± 0.0000 | 4.15 ± 0.00 | 20.80 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 2.1477 | 4.4732 | 3.44 |
| MTGNN | 1 | 2.00 | 2.2812 | 4.8506 | 3.65 |
| GraphWaveNet | 1 | 3.00 | 2.5944 | 4.9512 | 4.15 |

## Configurações Selecionadas

### simple_pems-bay_STICformer_22_04_2026-09h_29-epoch_1

- Modelo: STICformer
- Dataset: pems-bay
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.3267
- Teste (MAE): 2.1477
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_pems-bay_MTGNN_22_04_2026-09h_29-epoch_1

- Modelo: MTGNN
- Dataset: pems-bay
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.4993
- Teste (MAE): 2.2812
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_pems-bay_GraphWaveNet_22_04_2026-09h_29-epoch_1

- Modelo: GraphWaveNet
- Dataset: pems-bay
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.7716
- Teste (MAE): 2.5944
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_pems-bay_STICformer_22_04_2026-09h_29-epoch_1 | STICformer | pems-bay | 1 | 2.1477 | 4.4732 | 4.31 | 3.44 | 1 |
| simple_pems-bay_MTGNN_22_04_2026-09h_29-epoch_1 | MTGNN | pems-bay | 1 | 2.2812 | 4.8506 | 4.62 | 3.65 | 2 |
| simple_pems-bay_GraphWaveNet_22_04_2026-09h_29-epoch_1 | GraphWaveNet | pems-bay | 1 | 2.5944 | 4.9512 | 5.21 | 4.15 | 3 |
