# Relatório de Seleção de Configurações

**Gerado em:** 2026-04-08 07:26:47

**Total de experimentos consolidados:** 6

Os números abaixo representam o melhor conjunto de hiperparâmetros escolhido pela métrica de validação, sem executar a fase final de teste.

## Ranking por Dataset

### metr-la

| Rank | Modelo | Val MAE | Val RMSE | Delta vs melhor (%) |
|------|--------|---------|----------|----------------------|
| 1 | GraphWaveNet | 5.5038 ± 0.0000 | 11.4986 ± 0.0000 | 0.00 |
| 2 | DGCRN | 5.5709 ± 0.0000 | 11.4079 ± 0.0000 | 1.22 |
| 3 | MTGNN | 5.6909 ± 0.0000 | 11.2124 ± 0.0000 | 3.40 |
| 4 | STICformer | 5.6912 ± 0.0000 | 11.0389 ± 0.0000 | 3.40 |
| 5 | PatchSTG | 6.0227 ± 0.0000 | 11.0184 ± 0.0000 | 9.43 |
| 6 | DCRNN | 6.5807 ± 0.0000 | 11.8885 ± 0.0000 | 19.57 |

## Configurações Selecionadas

### original_metr-la_GraphWaveNet_08_04_2026-00h_18-epoch_2

- Modelo: GraphWaveNet
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds concluídas no search: 1
- Validação (MAE): 5.5038
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "epochs": 2, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "k": 2, "lr": 0.001, "num_blocks": 3, "output_dim": 1, "patience": 5, "seq_len": 12, "weight_decay": 0.0001}`

### original_metr-la_DGCRN_08_04_2026-00h_18-epoch_2

- Modelo: DGCRN
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds concluídas no search: 1
- Validação (MAE): 5.5709
- Parâmetros: `{"dropout": 0.1, "epochs": 2, "gcn_depth": 2, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "lr": 0.001, "node_dim": 16, "output_dim": 1, "patience": 5, "seq_len": 12, "weight_decay": 0.0001}`

### original_metr-la_MTGNN_08_04_2026-00h_18-epoch_2

- Modelo: MTGNN
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds concluídas no search: 1
- Validação (MAE): 5.6909
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "epochs": 2, "gcn_depth": 2, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "output_dim": 1, "patience": 5, "propalpha": 0.05, "seq_len": 12, "weight_decay": 0.0001}`

### original_metr-la_STICformer_08_04_2026-00h_18-epoch_2

- Modelo: STICformer
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds concluídas no search: 1
- Validação (MAE): 5.6912
- Parâmetros: `{"dropout": 0.1, "epochs": 2, "ff_multiplier": 2, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "lr": 0.001, "num_heads": 4, "num_layers": 2, "output_dim": 1, "patience": 5, "seq_len": 12, "weight_decay": 0.0001}`

### original_metr-la_PatchSTG_08_04_2026-00h_18-epoch_2

- Modelo: PatchSTG
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds concluídas no search: 1
- Validação (MAE): 6.0227
- Parâmetros: `{"dropout": 0.1, "epochs": 2, "ff_multiplier": 2, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "lr": 0.001, "num_heads": 4, "num_layers": 2, "output_dim": 1, "patch_len": 4, "patch_stride": 2, "patience": 5, "seq_len": 12, "weight_decay": 0.0001}`

### original_metr-la_DCRNN_08_04_2026-00h_18-epoch_2

- Modelo: DCRNN
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds concluídas no search: 1
- Validação (MAE): 6.5807
- Parâmetros: `{"dropout": 0.1, "epochs": 2, "hidden_dim": 64, "horizon": 12, "input_dim": 1, "k": 2, "lr": 0.001, "output_dim": 1, "patience": 5, "seq_len": 12, "teacher_forcing_ratio": 0.5, "use_scheduled_sampling": false, "weight_decay": 0.0001}`

