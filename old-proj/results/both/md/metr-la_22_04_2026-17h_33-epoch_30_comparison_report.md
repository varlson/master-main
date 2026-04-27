# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 19:40:52

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### metr-la

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 5.9972 ± 0.0000 | 11.8574 ± 0.0000 | 11.82 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 6.1061 ± 0.0000 | 12.0270 ± 0.0000 | 12.03 ± 0.00 | 1.82 |
| 3 | MTGNN | 6.8111 ± 0.0000 | 12.3707 ± 0.0000 | 13.42 ± 0.00 | 13.57 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 5.9972 | 11.8574 | 11.82 |
| GraphWaveNet | 1 | 2.00 | 6.1061 | 12.0270 | 12.03 |
| MTGNN | 1 | 3.00 | 6.8111 | 12.3707 | 13.42 |

## Configurações Selecionadas

### simple_metr-la_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.1875
- Teste (MAE): 5.9972
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_metr-la_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.3883
- Teste (MAE): 6.1061
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_metr-la_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.8495
- Teste (MAE): 6.8111
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_metr-la_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | metr-la | 1 | 5.9972 | 11.8574 | 34.23 | 11.82 | 1 |
| simple_metr-la_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | metr-la | 1 | 6.1061 | 12.0270 | 34.65 | 12.03 | 2 |
| simple_metr-la_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | metr-la | 1 | 6.8111 | 12.3707 | 35.20 | 13.42 | 3 |
