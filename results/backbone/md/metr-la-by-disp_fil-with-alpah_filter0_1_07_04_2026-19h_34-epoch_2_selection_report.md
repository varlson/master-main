# Relatório de Seleção de Configurações

**Gerado em:** 2026-04-07 20:40:12

**Total de experimentos consolidados:** 6

Os números abaixo representam o melhor conjunto de hiperparâmetros escolhido pela métrica de validação, sem executar a fase final de teste.

## Ranking por Dataset

### metr-la-by-disp_fil-with-alpah_filter0_1

| Rank | Modelo | Val MAE | Val RMSE | Delta vs melhor (%) |
|------|--------|---------|----------|----------------------|
| 1 | STICformer | 5.3292 ± 0.0000 | 10.8328 ± 0.0000 | 0.00 |
| 2 | GraphWaveNet | 5.5763 ± 0.0000 | 11.0582 ± 0.0000 | 4.64 |
| 3 | PatchSTG | 5.5991 ± 0.0000 | 10.8986 ± 0.0000 | 5.07 |
| 4 | DGCRN | 5.6250 ± 0.0000 | 11.1913 ± 0.0000 | 5.55 |
| 5 | MTGNN | 5.6875 ± 0.0000 | 11.1016 ± 0.0000 | 6.72 |
| 6 | DCRNN | 6.0986 ± 0.0000 | 11.5745 ± 0.0000 | 14.44 |

## Configurações Selecionadas

### backbone_metr-la-by-disp_fil-with-alpah_filter0_1_STICformer_07_04_2026-19h_34-epoch_2

- Modelo: STICformer
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_1
- Métrica de seleção: `val_mae`
- Seeds concluídas no search: 1
- Validação (MAE): 5.3292
- Parâmetros: `{"dropout": 0.1, "epochs": 2, "ff_multiplier": 2, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "lr": 0.001, "num_heads": 4, "num_layers": 2, "output_dim": 1, "patience": 5, "seq_len": 12, "weight_decay": 0.0001}`

### backbone_metr-la-by-disp_fil-with-alpah_filter0_1_GraphWaveNet_07_04_2026-19h_34-epoch_2

- Modelo: GraphWaveNet
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_1
- Métrica de seleção: `val_mae`
- Seeds concluídas no search: 1
- Validação (MAE): 5.5763
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "epochs": 2, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "k": 2, "lr": 0.001, "num_blocks": 3, "output_dim": 1, "patience": 5, "seq_len": 12, "weight_decay": 0.0001}`

### backbone_metr-la-by-disp_fil-with-alpah_filter0_1_PatchSTG_07_04_2026-19h_34-epoch_2

- Modelo: PatchSTG
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_1
- Métrica de seleção: `val_mae`
- Seeds concluídas no search: 1
- Validação (MAE): 5.5991
- Parâmetros: `{"dropout": 0.1, "epochs": 2, "ff_multiplier": 2, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "lr": 0.001, "num_heads": 4, "num_layers": 2, "output_dim": 1, "patch_len": 4, "patch_stride": 2, "patience": 5, "seq_len": 12, "weight_decay": 0.0001}`

### backbone_metr-la-by-disp_fil-with-alpah_filter0_1_DGCRN_07_04_2026-19h_34-epoch_2

- Modelo: DGCRN
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_1
- Métrica de seleção: `val_mae`
- Seeds concluídas no search: 1
- Validação (MAE): 5.6250
- Parâmetros: `{"dropout": 0.1, "epochs": 2, "gcn_depth": 2, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "lr": 0.001, "node_dim": 16, "output_dim": 1, "patience": 5, "seq_len": 12, "weight_decay": 0.0001}`

### backbone_metr-la-by-disp_fil-with-alpah_filter0_1_MTGNN_07_04_2026-19h_34-epoch_2

- Modelo: MTGNN
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_1
- Métrica de seleção: `val_mae`
- Seeds concluídas no search: 1
- Validação (MAE): 5.6875
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "epochs": 2, "gcn_depth": 2, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "output_dim": 1, "patience": 5, "propalpha": 0.05, "seq_len": 12, "weight_decay": 0.0001}`

### backbone_metr-la-by-disp_fil-with-alpah_filter0_1_DCRNN_07_04_2026-19h_34-epoch_2

- Modelo: DCRNN
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_1
- Métrica de seleção: `val_mae`
- Seeds concluídas no search: 1
- Validação (MAE): 6.0986
- Parâmetros: `{"dropout": 0.1, "epochs": 2, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "k": 2, "lr": 0.001, "output_dim": 1, "patience": 5, "seq_len": 12, "teacher_forcing_ratio": 0.5, "use_scheduled_sampling": false, "weight_decay": 0.0001}`

