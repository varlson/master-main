# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 13:03:21

**Total de experimentos consolidados:** 18

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### metr-la-by-disp_fil-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 5.9449 ± 0.0000 | 12.0806 ± 0.0000 | 11.72 ± 0.00 | 0.00 |
| 2 | MTGNN | 6.8223 ± 0.0000 | 12.3648 ± 0.0000 | 13.45 ± 0.00 | 14.76 |
| 3 | GraphWaveNet | 7.0205 ± 0.0000 | 12.3617 ± 0.0000 | 13.84 ± 0.00 | 18.09 |

### metr-la-by-high_sal-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 6.4325 ± 0.0000 | 11.9879 ± 0.0000 | 12.64 ± 0.00 | 0.00 |
| 2 | MTGNN | 6.6687 ± 0.0000 | 12.2793 ± 0.0000 | 13.11 ± 0.00 | 3.67 |
| 3 | GraphWaveNet | 7.1818 ± 0.0000 | 12.4138 ± 0.0000 | 14.12 ± 0.00 | 11.65 |

### metr-la-by-nois_corr-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 5.9981 ± 0.0000 | 12.0306 ± 0.0000 | 11.82 ± 0.00 | 0.00 |
| 2 | MTGNN | 6.8843 ± 0.0000 | 12.3703 ± 0.0000 | 13.57 ± 0.00 | 14.78 |
| 3 | GraphWaveNet | 7.4615 ± 0.0000 | 12.4822 ± 0.0000 | 14.71 ± 0.00 | 24.40 |

### pems-bay-by-disp_fil-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 2.1766 ± 0.0000 | 4.4707 ± 0.0000 | 3.48 ± 0.00 | 0.00 |
| 2 | MTGNN | 2.2750 ± 0.0000 | 4.8442 ± 0.0000 | 3.64 ± 0.00 | 4.52 |
| 3 | GraphWaveNet | 2.4358 ± 0.0000 | 5.0032 ± 0.0000 | 3.90 ± 0.00 | 11.91 |

### pems-bay-by-high_sal-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 2.0089 ± 0.0000 | 4.3699 ± 0.0000 | 3.21 ± 0.00 | 0.00 |
| 2 | MTGNN | 2.2242 ± 0.0000 | 4.7851 ± 0.0000 | 3.55 ± 0.00 | 10.72 |
| 3 | GraphWaveNet | 2.5068 ± 0.0000 | 5.0574 ± 0.0000 | 4.00 ± 0.00 | 24.78 |

### pems-bay-by-nois_corr-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 2.1643 ± 0.0000 | 4.4357 ± 0.0000 | 3.47 ± 0.00 | 0.00 |
| 2 | MTGNN | 2.2685 ± 0.0000 | 4.8414 ± 0.0000 | 3.63 ± 0.00 | 4.81 |
| 3 | GraphWaveNet | 2.5557 ± 0.0000 | 5.1512 ± 0.0000 | 4.09 ± 0.00 | 18.08 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 6 | 1.00 | 4.1209 | 8.2292 | 7.72 |
| MTGNN | 6 | 2.00 | 4.5238 | 8.5808 | 8.49 |
| GraphWaveNet | 6 | 3.00 | 4.8604 | 8.7449 | 9.11 |

## Configurações Selecionadas

### simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1

- Modelo: STICformer
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.2785
- Teste (MAE): 5.9449
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1

- Modelo: MTGNN
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.8859
- Teste (MAE): 6.8223
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1

- Modelo: GraphWaveNet
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 6.1700
- Teste (MAE): 7.0205
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

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

### simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1

- Modelo: STICformer
- Dataset: metr-la-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.3035
- Teste (MAE): 5.9981
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1

- Modelo: MTGNN
- Dataset: metr-la-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.9360
- Teste (MAE): 6.8843
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1

- Modelo: GraphWaveNet
- Dataset: metr-la-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 6.5518
- Teste (MAE): 7.4615
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1

- Modelo: STICformer
- Dataset: pems-bay-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.3483
- Teste (MAE): 2.1766
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1

- Modelo: MTGNN
- Dataset: pems-bay-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.4933
- Teste (MAE): 2.2750
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1

- Modelo: GraphWaveNet
- Dataset: pems-bay-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.6449
- Teste (MAE): 2.4358
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

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

### simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1

- Modelo: STICformer
- Dataset: pems-bay-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.3343
- Teste (MAE): 2.1643
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1

- Modelo: MTGNN
- Dataset: pems-bay-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.4856
- Teste (MAE): 2.2685
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1

- Modelo: GraphWaveNet
- Dataset: pems-bay-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.7675
- Teste (MAE): 2.5557
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1 | STICformer | metr-la-by-disp_fil-with-alpah_filter0_4 | 1 | 5.9449 | 12.0806 | 34.67 | 11.72 | 1 |
| simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1 | MTGNN | metr-la-by-disp_fil-with-alpah_filter0_4 | 1 | 6.8223 | 12.3648 | 35.24 | 13.45 | 2 |
| simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1 | GraphWaveNet | metr-la-by-disp_fil-with-alpah_filter0_4 | 1 | 7.0205 | 12.3617 | 35.41 | 13.84 | 3 |
| simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1 | STICformer | metr-la-by-high_sal-with-alpah_filter0_4 | 1 | 6.4325 | 11.9879 | 34.38 | 12.64 | 1 |
| simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1 | MTGNN | metr-la-by-high_sal-with-alpah_filter0_4 | 1 | 6.6687 | 12.2793 | 34.82 | 13.11 | 2 |
| simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1 | GraphWaveNet | metr-la-by-high_sal-with-alpah_filter0_4 | 1 | 7.1818 | 12.4138 | 35.50 | 14.12 | 3 |
| simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1 | STICformer | metr-la-by-nois_corr-with-alpah_filter0_4 | 1 | 5.9981 | 12.0306 | 34.59 | 11.82 | 1 |
| simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1 | MTGNN | metr-la-by-nois_corr-with-alpah_filter0_4 | 1 | 6.8843 | 12.3703 | 35.27 | 13.57 | 2 |
| simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1 | GraphWaveNet | metr-la-by-nois_corr-with-alpah_filter0_4 | 1 | 7.4615 | 12.4822 | 35.74 | 14.71 | 3 |
| simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1 | STICformer | pems-bay-by-disp_fil-with-alpah_filter0_4 | 1 | 2.1766 | 4.4707 | 4.34 | 3.48 | 1 |
| simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1 | MTGNN | pems-bay-by-disp_fil-with-alpah_filter0_4 | 1 | 2.2750 | 4.8442 | 4.61 | 3.64 | 2 |
| simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1 | GraphWaveNet | pems-bay-by-disp_fil-with-alpah_filter0_4 | 1 | 2.4358 | 5.0032 | 5.02 | 3.90 | 3 |
| simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1 | STICformer | pems-bay-by-high_sal-with-alpah_filter0_4 | 1 | 2.0089 | 4.3699 | 4.07 | 3.21 | 1 |
| simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1 | MTGNN | pems-bay-by-high_sal-with-alpah_filter0_4 | 1 | 2.2242 | 4.7851 | 4.53 | 3.55 | 2 |
| simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1 | GraphWaveNet | pems-bay-by-high_sal-with-alpah_filter0_4 | 1 | 2.5068 | 5.0574 | 5.23 | 4.00 | 3 |
| simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1 | STICformer | pems-bay-by-nois_corr-with-alpah_filter0_4 | 1 | 2.1643 | 4.4357 | 4.32 | 3.47 | 1 |
| simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1 | MTGNN | pems-bay-by-nois_corr-with-alpah_filter0_4 | 1 | 2.2685 | 4.8414 | 4.60 | 3.63 | 2 |
| simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1 | GraphWaveNet | pems-bay-by-nois_corr-with-alpah_filter0_4 | 1 | 2.5557 | 5.1512 | 5.36 | 4.09 | 3 |
