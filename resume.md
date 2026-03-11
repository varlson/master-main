# Relatório Resumido do Projeto (run `20260304_163952`)

## 1) Objetivo e estado atual
O objetivo do projeto é comparar modelos espaço-temporais para previsão de tráfego em grafo (METR-LA e PEMS-BAY) em um pipeline único de treino/validação/teste e consolidação de resultados.

Estado atual (com base em `results/md/all-datasets_20260304_163952_comparison_report.md`):
- 12 experimentos concluídos (6 modelos x 2 datasets).
- Melhor em `pems-bay`: **STICformer** (`MAE=0.2014`, `RMSE=0.4331`).
- Melhor em `metr-la` por MAE: **DGCRN** (`MAE=0.2757`), mas com sinais de maior instabilidade de validação.

## 2) Modelos testados (quando surgiram e por que fazem sentido)
Linha temporal aproximada da literatura de forecasting em grafos:

- **DCRNN (2018, geração RNN+difusão)**: forte baseline para dinâmica de tráfego e dependência temporal sequencial.
- **GraphWaveNet (2019, geração convolucional espaço-temporal)**: bom custo-benefício, normalmente estável e competitivo.
- **MTGNN (2020, geração com grafo aprendido dinamicamente)**: útil para capturar relações entre nós que não aparecem bem na adjacência fixa.
- **DGCRN (2021, geração dinâmica com recorrência em grafo)**: tende a capturar mudanças temporais de conectividade; vale testar para cenários não estacionários.
- **STICformer (mais recente, família Transformer espaço-temporal)**: forte para dependências de médio/longo alcance e bom desempenho em benchmarks recentes.
- **PatchSTG (mais recente, família Transformer por patches)**: melhora eficiência e costuma generalizar bem em séries longas.

Para seu objetivo (comparar arquiteturas em tráfego), o conjunto está bem montado: cobre baseline clássico, convolucional, recorrente dinâmica e duas abordagens Transformer modernas.

## 3) Resumo rápido por dataset
Fonte principal: `results/md/all-datasets_20260304_163952_comparison_report.md`.

### METR-LA (ranking por MAE)
1. DGCRN (`0.2757`)
2. PatchSTG (`0.2937`)
3. STICformer (`0.3048`)
4. GraphWaveNet (`0.3139`)
5. DCRNN (`0.3445`)
6. MTGNN (`0.3468`)

### PEMS-BAY (ranking por MAE)
1. STICformer (`0.2014`)
2. PatchSTG (`0.2083`)
3. GraphWaveNet (`0.2244`)
4. MTGNN (`0.2359`)
5. DGCRN (`0.2696`)
6. DCRNN (`0.2841`)

## 4) Leitura dos plots por modelo e dataset
### Convenções usadas nesta seção
- `metrics_by_horizon.png`: mostrei a degradação de curto -> longo prazo (h1 -> h12).
- `scatter_real_vs_pred.png`: usei os `R²` exibidos no título dos subplots (nós mais difíceis).
- `train_val_curves.png`: foco em convergência e gap treino-validação.

### A) `metr-la`

- **DCRNN**
  - `metrics_by_horizon`: degradação mais lenta que os demais (`MAE +76.7%`, `RMSE +69.9%`), porém parte de erro alto.
  - `scatter_real_vs_pred`: muito heterogêneo (`R²` de `0.009` a `0.864`), com nós problemáticos claros.
  - `train_val_curves`: converge, mas com gap estável (~`0.03-0.04`) e val em platô; generalização mediana.

- **GraphWaveNet**
  - `metrics_by_horizon`: erro cresce forte com horizonte (`MAE +153.9%`, `RMSE +107.7%`).
  - `scatter_real_vs_pred`: consistente (`R²` ~`0.78-0.85`) sem colapso extremo em nó específico.
  - `train_val_curves`: curva estável, val reduz com oscilações leves; bom equilíbrio.

- **MTGNN**
  - `metrics_by_horizon`: degradação forte (`MAE +148.8%`, `RMSE +123.3%`).
  - `scatter_real_vs_pred`: bom ajuste global (`R²` ~`0.82-0.90`).
  - `train_val_curves`: poucas épocas (parada precoce), val volta a subir após melhora inicial; sinal de limite de generalização.

- **DGCRN**
  - `metrics_by_horizon`: melhor MAE global no dataset, mas com degradação forte no horizonte (`MAE +154.0%`, `RMSE +134.5%`).
  - `scatter_real_vs_pred`: sólido nos nós difíceis (`R²` ~`0.84-0.90`).
  - `train_val_curves`: gap muito alto (train muito baixo e val oscilante ~`0.33-0.40`); maior risco de overfitting.

- **STICformer**
  - `metrics_by_horizon`: degradação forte, mas menos agressiva que vários pares (`MAE +115.0%`, `RMSE +112.4%`).
  - `scatter_real_vs_pred`: forte consistência (`R²` ~`0.84-0.89`).
  - `train_val_curves`: treino cai continuamente; val oscila em faixa estreita, com leve subida no fim.

- **PatchSTG**
  - `metrics_by_horizon`: desempenho global bom, mas degradação relevante (`MAE +153.3%`, `RMSE +121.1%`).
  - `scatter_real_vs_pred`: consistente (`R²` ~`0.83-0.89`), semelhante ao STICformer.
  - `train_val_curves`: convergência boa; val relativamente estável com pequeno rebound final.

Leitura geral do `metr-la`: **DGCRN ganha em MAE**, mas **PatchSTG/STICformer** mostram perfil mais equilibrado entre ranking e estabilidade visual.

### B) `pems-bay`

- **DCRNN**
  - `metrics_by_horizon`: menor degradação relativa (`MAE +25.8%`, `RMSE +43.8%`), mas com pior erro absoluto.
  - `scatter_real_vs_pred`: fraco nos nós difíceis (`R²` de `-2.71` a `0.65`), com dispersão alta.
  - `train_val_curves`: gap alto e persistente; overfitting claro.

- **GraphWaveNet**
  - `metrics_by_horizon`: degradação forte (`MAE +126.3%`, `RMSE +146.3%`).
  - `scatter_real_vs_pred`: muito bom em 5/6 nós (`R²` alto), mas 1 nó crítico com `R²~0.25`.
  - `train_val_curves`: convergência estável, val cai de forma gradual; bom comportamento geral.

- **MTGNN**
  - `metrics_by_horizon`: degradação forte (`MAE +196.8%`, `RMSE +222.8%`).
  - `scatter_real_vs_pred`: padrão semelhante ao GraphWaveNet (maioria boa, 1 nó difícil com `R²~0.33`).
  - `train_val_curves`: train cai, mas val quase lateral; gap relativamente alto.

- **DGCRN**
  - `metrics_by_horizon`: pior degradação no dataset (`MAE +328.3%`, `RMSE +274.2%`).
  - `scatter_real_vs_pred`: heterogêneo (de `R²~0.37` até `0.96`), bom em parte dos nós, fraco em outros.
  - `train_val_curves`: poucas épocas e gap extremo (train muito baixo vs val alta); forte risco de sobreajuste.

- **STICformer**
  - `metrics_by_horizon`: melhor resultado absoluto no dataset; degradação de horizonte existe (`MAE +142.3%`, `RMSE +181.9%`).
  - `scatter_real_vs_pred`: forte em quase todos os nós (`R²` alto, com 1 nó difícil em `~0.46`).
  - `train_val_curves`: melhor compromisso visual entre queda de val e gap final entre modelos top.

- **PatchSTG**
  - `metrics_by_horizon`: 2º melhor global, com degradação parecida ao STICformer (`MAE +157.1%`, `RMSE +188.8%`).
  - `scatter_real_vs_pred`: excelente em maioria dos nós (`R²` alto), 1 nó difícil (`~0.39`).
  - `train_val_curves`: convergência boa e estável; gap moderado.

Leitura geral do `pems-bay`: **STICformer e PatchSTG** são as opções mais fortes e mais equilibradas; GraphWaveNet vem logo atrás com comportamento robusto.

## 5) Conclusões práticas
- Para escolha principal hoje, com base nos artefatos atuais: **STICformer** (principal) e **PatchSTG** (backup forte).
- Para `metr-la`, vale manter **DGCRN** como candidato por MAE, mas com atenção ao risco de overfitting observado em `train_val_curves`.
- Os `MAPE` por horizonte estão muito altos/instáveis em parte dos gráficos (principalmente com valores próximos de zero no denominador), então a decisão deve priorizar **MAE/RMSE + leitura de scatter**.

## 6) Arquivos usados como base
- `results/md/all-datasets_20260304_163952_comparison_report.md`
- `results/md/metr-la_20260304_163952_comparison_report.md`
- `results/md/pems-bay_20260304_163952_comparison_report.md`
- `results/plots/*/metrics_by_horizon.png`
- `results/plots/*/scatter_real_vs_pred.png`
- `results/plots/*/train_val_curves.png`
