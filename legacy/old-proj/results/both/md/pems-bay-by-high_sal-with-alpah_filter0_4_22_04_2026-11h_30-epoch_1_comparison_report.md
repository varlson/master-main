# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 13:03:21

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### pems-bay-by-high_sal-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 2.0089 ± 0.0000 | 4.3699 ± 0.0000 | 3.21 ± 0.00 | 0.00 |
| 2 | MTGNN | 2.2242 ± 0.0000 | 4.7851 ± 0.0000 | 3.55 ± 0.00 | 10.72 |
| 3 | GraphWaveNet | 2.5068 ± 0.0000 | 5.0574 ± 0.0000 | 4.00 ± 0.00 | 24.78 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 2.0089 | 4.3699 | 3.21 |
| MTGNN | 1 | 2.00 | 2.2242 | 4.7851 | 3.55 |
| GraphWaveNet | 1 | 3.00 | 2.5068 | 5.0574 | 4.00 |

## Configurações Selecionadas

### simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1

- Modelo: STICformer
- Dataset: pems-bay-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.1857
- Teste (MAE): 2.0089
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1

- Modelo: MTGNN
- Dataset: pems-bay-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.4515
- Teste (MAE): 2.2242
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1

- Modelo: GraphWaveNet
- Dataset: pems-bay-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.7110
- Teste (MAE): 2.5068
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1 | STICformer | pems-bay-by-high_sal-with-alpah_filter0_4 | 1 | 2.0089 | 4.3699 | 4.07 | 3.21 | 1 |
| simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1 | MTGNN | pems-bay-by-high_sal-with-alpah_filter0_4 | 1 | 2.2242 | 4.7851 | 4.53 | 3.55 | 2 |
| simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1 | GraphWaveNet | pems-bay-by-high_sal-with-alpah_filter0_4 | 1 | 2.5068 | 5.0574 | 5.23 | 4.00 | 3 |
