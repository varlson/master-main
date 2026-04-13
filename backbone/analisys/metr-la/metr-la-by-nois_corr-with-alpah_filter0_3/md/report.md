# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-09 00:21:04

Pasta de saida: `/home/varlson/master/main/master-main/backbone/analisys/metr-la/metr-la-by-nois_corr-with-alpah_filter0_3`

## Dataset: metr-la

### Rede Original

- Nos: 207 | Arestas: 1313 | Densidade: 0.0616
- Grau medio: 12.6860 | Forca media: 5.1448
- Clustering medio: 0.5485 | Eficiencia global: 0.2789 | Modularidade: 0.6978
- Caminho medio (gcc): 5.0637 | Componentes: 2 | Assortatividade: 0.4616

- Comunidades: 8 | Maior comunidade: 53 | Razao do componente gigante: 0.9952

### Comparacoes com a Rede Original

#### metr-la-by-nois_corr-with-alpah_filter0_3

Informacoes basicas do backbone:
- Nos: 207 | Delta: -0.00%
- Arestas: 1022 | Delta: -22.16%
- Densidade: 0.0479 | Delta: -22.16%
- Grau medio: 9.8744 | Forca media: 4.6732
- Componentes: 2 | Razao do componente gigante: 0.9952 | Delta GCC: 0.00%
- Clustering medio: 0.4829 | Delta: -11.97%
- Eficiencia global: 0.2545 | Delta: -8.73%
- Caminho medio (gcc): 5.5292 | Delta: +9.19%
- Modularidade: 0.7094 | Delta: +0.0116
- Assortatividade: 0.2622 | Delta: -0.1994
- Comunidades: 8 | Maior comunidade: 53
- Robustez aleatoria (AUC LCC): 0.4363 | Delta: -4.28%
- Robustez alvo (AUC LCC): 0.3553 | Delta: -8.35%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.9630
- Menor correlacao de centralidade entre as metricas: 0.8444
- Overlap medio top-k: 0.7700
- Leitura interpretativa:
  - reduziu a conectividade de forma moderada, preservando parte relevante das arestas
  - preservou bem o ranking relativo das centralidades

