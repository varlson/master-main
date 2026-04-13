# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-09 00:20:56

Pasta de saida: `/home/varlson/master/main/master-main/backbone/analisys/metr-la/metr-la-by-disp_fil-with-alpah_filter0_3`

## Dataset: metr-la

### Rede Original

- Nos: 207 | Arestas: 1313 | Densidade: 0.0616
- Grau medio: 12.6860 | Forca media: 5.1448
- Clustering medio: 0.5485 | Eficiencia global: 0.2789 | Modularidade: 0.6978
- Caminho medio (gcc): 5.0637 | Componentes: 2 | Assortatividade: 0.4616

- Comunidades: 8 | Maior comunidade: 53 | Razao do componente gigante: 0.9952

### Comparacoes com a Rede Original

#### metr-la-by-disp_fil-with-alpah_filter0_3

Informacoes basicas do backbone:
- Nos: 207 | Delta: -0.00%
- Arestas: 364 | Delta: -72.28%
- Densidade: 0.0171 | Delta: -72.28%
- Grau medio: 3.5169 | Forca media: 2.7281
- Componentes: 17 | Razao do componente gigante: 0.8454 | Delta GCC: -15.05%
- Clustering medio: 0.3913 | Delta: -28.65%
- Eficiencia global: 0.1072 | Delta: -61.57%
- Caminho medio (gcc): 10.7903 | Delta: +113.09%
- Modularidade: 0.8314 | Delta: +0.1336
- Assortatividade: 0.4800 | Delta: +0.0183
- Comunidades: 28 | Maior comunidade: 24
- Robustez aleatoria (AUC LCC): 0.2092 | Delta: -54.10%
- Robustez alvo (AUC LCC): 0.1095 | Delta: -71.74%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.7633
- Menor correlacao de centralidade entre as metricas: 0.5883
- Overlap medio top-k: 0.6600
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - ficou mais frágil a ataques direcionados em hubs

