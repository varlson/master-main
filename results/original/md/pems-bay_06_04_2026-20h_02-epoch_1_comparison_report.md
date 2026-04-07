# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-06 21:22:50

**Total de experimentos consolidados:** 1

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### pems-bay

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | GraphWaveNet | 2.2694 ± 0.0000 | 4.6013 ± 0.0000 | 3.63 ± 0.00 | 0.00 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| GraphWaveNet | 1 | 1.00 | 2.2694 | 4.6013 | 3.63 |

## Configurações Selecionadas

### original_pems-bay_GraphWaveNet_06_04_2026-20h_02-epoch_1

- Modelo: GraphWaveNet
- Dataset: pems-bay
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.4542
- Teste (MAE): 2.2694
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "epochs": 1, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "k": 2, "lr": 0.001, "num_blocks": 3, "output_dim": 1, "patience": 5, "seq_len": 12, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| original_pems-bay_GraphWaveNet_06_04_2026-20h_02-epoch_1 | GraphWaveNet | pems-bay | 1 | 2.2694 | 4.6013 | 4.58 | 3.63 | 1 |
