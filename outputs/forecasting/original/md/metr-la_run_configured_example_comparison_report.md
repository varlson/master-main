# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-29 21:47:22

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### metr-la

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | GraphWaveNet | 6.3357 ± 0.0000 | 12.4769 ± 0.0000 | 12.49 ± 0.00 | 0.00 |
| 2 | DGCRN | 6.5017 ± 0.0000 | 12.6292 ± 0.0000 | 12.81 ± 0.00 | 2.62 |
| 3 | MTGNN | 6.6695 ± 0.0000 | 12.3112 ± 0.0000 | 13.14 ± 0.00 | 5.27 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| GraphWaveNet | 1 | 1.00 | 6.3357 | 12.4769 | 12.49 |
| DGCRN | 1 | 2.00 | 6.5017 | 12.6292 | 12.81 |
| MTGNN | 1 | 3.00 | 6.6695 | 12.3112 | 13.14 |

## Configurações Selecionadas

### original_metr-la_GraphWaveNet_run_configured_example

- Modelo: GraphWaveNet
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): nan
- Teste (MAE): 6.3357
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "epochs": 2, "hidden_dim": 32, "horizon": 12, "input_dim": 1, "k": 2, "lr": 0.001, "num_blocks": 2, "output_dim": 1, "patience": 5, "seq_len": 12, "weight_decay": 0.0001}`

### original_metr-la_DGCRN_run_configured_example

- Modelo: DGCRN
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): nan
- Teste (MAE): 6.5017
- Parâmetros: `{"dropout": 0.1, "epochs": 2, "gcn_depth": 2, "hidden_dim": 32, "horizon": 12, "input_dim": 1, "lr": 0.001, "node_dim": 16, "output_dim": 1, "patience": 5, "seq_len": 12, "weight_decay": 0.0001}`

### original_metr-la_MTGNN_run_configured_example

- Modelo: MTGNN
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): nan
- Teste (MAE): 6.6695
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "epochs": 2, "gcn_depth": 2, "hidden_dim": 32, "horizon": 12, "input_dim": 1, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 2, "output_dim": 1, "patience": 5, "propalpha": 0.05, "seq_len": 12, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| original_metr-la_GraphWaveNet_run_configured_example | GraphWaveNet | metr-la | 1 | 6.3357 | 12.4769 | 34.87 | 12.49 | 1 |
| original_metr-la_DGCRN_run_configured_example | DGCRN | metr-la | 1 | 6.5017 | 12.6292 | 36.65 | 12.81 | 2 |
| original_metr-la_MTGNN_run_configured_example | MTGNN | metr-la | 1 | 6.6695 | 12.3112 | 35.42 | 13.14 | 3 |
