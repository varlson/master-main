# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 21:52:26

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### pems-bay-by-disp_fil-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 1.9605 ± 0.0000 | 4.1158 ± 0.0000 | 3.14 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 2.1188 ± 0.0000 | 4.4294 ± 0.0000 | 3.39 ± 0.00 | 8.08 |
| 3 | MTGNN | 2.1935 ± 0.0000 | 4.7404 ± 0.0000 | 3.51 ± 0.00 | 11.89 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 1.9605 | 4.1158 | 3.14 |
| GraphWaveNet | 1 | 2.00 | 2.1188 | 4.4294 | 3.39 |
| MTGNN | 1 | 3.00 | 2.1935 | 4.7404 | 3.51 |

## Configurações Selecionadas

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

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | pems-bay-by-disp_fil-with-alpah_filter0_4 | 1 | 1.9605 | 4.1158 | 3.95 | 3.14 | 1 |
| simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | pems-bay-by-disp_fil-with-alpah_filter0_4 | 1 | 2.1188 | 4.4294 | 4.28 | 3.39 | 2 |
| simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | pems-bay-by-disp_fil-with-alpah_filter0_4 | 1 | 2.1935 | 4.7404 | 4.46 | 3.51 | 3 |
