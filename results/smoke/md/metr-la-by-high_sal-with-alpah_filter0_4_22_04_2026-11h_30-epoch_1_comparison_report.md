# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 12:02:29

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### metr-la-by-high_sal-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 6.4325 ± 0.0000 | 11.9879 ± 0.0000 | 12.64 ± 0.00 | 0.00 |
| 2 | MTGNN | 6.6687 ± 0.0000 | 12.2793 ± 0.0000 | 13.11 ± 0.00 | 3.67 |
| 3 | GraphWaveNet | 7.1818 ± 0.0000 | 12.4138 ± 0.0000 | 14.12 ± 0.00 | 11.65 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 6.4325 | 11.9879 | 12.64 |
| MTGNN | 1 | 2.00 | 6.6687 | 12.2793 | 13.11 |
| GraphWaveNet | 1 | 3.00 | 7.1818 | 12.4138 | 14.12 |

## Configurações Selecionadas

### simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1

- Modelo: STICformer
- Dataset: metr-la-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.5734
- Teste (MAE): 6.4325
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1

- Modelo: MTGNN
- Dataset: metr-la-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.7125
- Teste (MAE): 6.6687
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1

- Modelo: GraphWaveNet
- Dataset: metr-la-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 6.3210
- Teste (MAE): 7.1818
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1 | STICformer | metr-la-by-high_sal-with-alpah_filter0_4 | 1 | 6.4325 | 11.9879 | 34.38 | 12.64 | 1 |
| simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1 | MTGNN | metr-la-by-high_sal-with-alpah_filter0_4 | 1 | 6.6687 | 12.2793 | 34.82 | 13.11 | 2 |
| simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1 | GraphWaveNet | metr-la-by-high_sal-with-alpah_filter0_4 | 1 | 7.1818 | 12.4138 | 35.50 | 14.12 | 3 |
