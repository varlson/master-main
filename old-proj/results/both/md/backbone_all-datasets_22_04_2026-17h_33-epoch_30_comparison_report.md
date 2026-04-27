# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-23 00:07:34

**Total de experimentos consolidados:** 18

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### metr-la-by-disp_fil-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 5.9656 ± 0.0000 | 11.8588 ± 0.0000 | 11.76 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 6.3341 ± 0.0000 | 12.0910 ± 0.0000 | 12.48 ± 0.00 | 6.18 |
| 3 | MTGNN | 6.8253 ± 0.0000 | 12.3840 ± 0.0000 | 13.45 ± 0.00 | 14.41 |

### metr-la-by-high_sal-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 6.0864 ± 0.0000 | 11.9126 ± 0.0000 | 11.96 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 6.2041 ± 0.0000 | 12.0241 ± 0.0000 | 12.20 ± 0.00 | 1.93 |
| 3 | MTGNN | 7.0226 ± 0.0000 | 12.3051 ± 0.0000 | 13.80 ± 0.00 | 15.38 |

### metr-la-by-nois_corr-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 5.9493 ± 0.0000 | 11.8476 ± 0.0000 | 11.73 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 6.1305 ± 0.0000 | 12.0339 ± 0.0000 | 12.08 ± 0.00 | 3.05 |
| 3 | MTGNN | 6.7817 ± 0.0000 | 12.3646 ± 0.0000 | 13.37 ± 0.00 | 13.99 |

### pems-bay-by-disp_fil-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 1.9605 ± 0.0000 | 4.1158 ± 0.0000 | 3.14 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 2.1188 ± 0.0000 | 4.4294 ± 0.0000 | 3.39 ± 0.00 | 8.08 |
| 3 | MTGNN | 2.1935 ± 0.0000 | 4.7404 ± 0.0000 | 3.51 ± 0.00 | 11.89 |

### pems-bay-by-high_sal-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 1.9132 ± 0.0000 | 4.1155 ± 0.0000 | 3.05 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 2.0694 ± 0.0000 | 4.2392 ± 0.0000 | 3.30 ± 0.00 | 8.16 |
| 3 | MTGNN | 2.1616 ± 0.0000 | 4.6703 ± 0.0000 | 3.45 ± 0.00 | 12.98 |

### pems-bay-by-nois_corr-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 1.8877 ± 0.0000 | 4.0166 ± 0.0000 | 3.02 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 2.0912 ± 0.0000 | 4.3888 ± 0.0000 | 3.35 ± 0.00 | 10.78 |
| 3 | MTGNN | 2.1868 ± 0.0000 | 4.6290 ± 0.0000 | 3.50 ± 0.00 | 15.84 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 6 | 1.00 | 3.9604 | 7.9778 | 7.44 |
| GraphWaveNet | 6 | 2.00 | 4.1580 | 8.2011 | 7.80 |
| MTGNN | 6 | 3.00 | 4.5286 | 8.5156 | 8.51 |

## Configurações Selecionadas

### simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.1705
- Teste (MAE): 5.9656
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.4720
- Teste (MAE): 6.3341
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.8570
- Teste (MAE): 6.8253
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: metr-la-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.1382
- Teste (MAE): 6.0864
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: metr-la-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.4267
- Teste (MAE): 6.2041
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: metr-la-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 6.0663
- Teste (MAE): 7.0226
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: metr-la-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.1292
- Teste (MAE): 5.9493
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: metr-la-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.3406
- Teste (MAE): 6.1305
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: metr-la-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.8245
- Teste (MAE): 6.7817
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: pems-bay-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.0789
- Teste (MAE): 1.9605
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: pems-bay-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.3066
- Teste (MAE): 2.1188
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: pems-bay-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.3614
- Teste (MAE): 2.1935
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: pems-bay-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.0332
- Teste (MAE): 1.9132
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: pems-bay-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.2017
- Teste (MAE): 2.0694
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: pems-bay-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.3280
- Teste (MAE): 2.1616
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: pems-bay-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 1.9917
- Teste (MAE): 1.8877
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: pems-bay-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.2679
- Teste (MAE): 2.0912
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: pems-bay-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.3355
- Teste (MAE): 2.1868
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | metr-la-by-disp_fil-with-alpah_filter0_4 | 1 | 5.9656 | 11.8588 | 34.12 | 11.76 | 1 |
| simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | metr-la-by-disp_fil-with-alpah_filter0_4 | 1 | 6.3341 | 12.0910 | 34.46 | 12.48 | 2 |
| simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | metr-la-by-disp_fil-with-alpah_filter0_4 | 1 | 6.8253 | 12.3840 | 35.16 | 13.45 | 3 |
| simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | metr-la-by-high_sal-with-alpah_filter0_4 | 1 | 6.0864 | 11.9126 | 33.65 | 11.96 | 1 |
| simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | metr-la-by-high_sal-with-alpah_filter0_4 | 1 | 6.2041 | 12.0241 | 34.58 | 12.20 | 2 |
| simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | metr-la-by-high_sal-with-alpah_filter0_4 | 1 | 7.0226 | 12.3051 | 35.29 | 13.80 | 3 |
| simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | metr-la-by-nois_corr-with-alpah_filter0_4 | 1 | 5.9493 | 11.8476 | 34.03 | 11.73 | 1 |
| simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | metr-la-by-nois_corr-with-alpah_filter0_4 | 1 | 6.1305 | 12.0339 | 34.41 | 12.08 | 2 |
| simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | metr-la-by-nois_corr-with-alpah_filter0_4 | 1 | 6.7817 | 12.3646 | 35.14 | 13.37 | 3 |
| simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | pems-bay-by-disp_fil-with-alpah_filter0_4 | 1 | 1.9605 | 4.1158 | 3.95 | 3.14 | 1 |
| simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | pems-bay-by-disp_fil-with-alpah_filter0_4 | 1 | 2.1188 | 4.4294 | 4.28 | 3.39 | 2 |
| simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | pems-bay-by-disp_fil-with-alpah_filter0_4 | 1 | 2.1935 | 4.7404 | 4.46 | 3.51 | 3 |
| simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | pems-bay-by-high_sal-with-alpah_filter0_4 | 1 | 1.9132 | 4.1155 | 3.84 | 3.05 | 1 |
| simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | pems-bay-by-high_sal-with-alpah_filter0_4 | 1 | 2.0694 | 4.2392 | 4.17 | 3.30 | 2 |
| simple_backbone_pems-bay-by-high_sal-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | pems-bay-by-high_sal-with-alpah_filter0_4 | 1 | 2.1616 | 4.6703 | 4.41 | 3.45 | 3 |
| simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | pems-bay-by-nois_corr-with-alpah_filter0_4 | 1 | 1.8877 | 4.0166 | 3.79 | 3.02 | 1 |
| simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | pems-bay-by-nois_corr-with-alpah_filter0_4 | 1 | 2.0912 | 4.3888 | 4.22 | 3.35 | 2 |
| simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | pems-bay-by-nois_corr-with-alpah_filter0_4 | 1 | 2.1868 | 4.6290 | 4.42 | 3.50 | 3 |
