# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-09 00:19:32

Pasta de saida: `/home/varlson/master/main/master-main/backbone/analisys/metr-la/metr-la-by-nois_corr-with-alpah_filter0_1`

## Dataset: metr-la

### Rede Original

- Nos: 207 | Arestas: 1313 | Densidade: 0.0616
- Grau medio: 12.6860 | Forca media: 5.1448
- Clustering medio: 0.5485 | Eficiencia global: 0.2789 | Modularidade: 0.6978
- Caminho medio (gcc): 5.0637 | Componentes: 2 | Assortatividade: 0.4616

- Comunidades: 8 | Maior comunidade: 53 | Razao do componente gigante: 0.9952

### Comparacoes com a Rede Original

#### metr-la-by-nois_corr-with-alpah_filter0_1

Informacoes basicas do backbone:
- Nos: 207 | Delta: -0.00%
- Arestas: 378 | Delta: -71.21%
- Densidade: 0.0177 | Delta: -71.21%
- Grau medio: 3.6522 | Forca media: 2.2599
- Componentes: 5 | Razao do componente gigante: 0.9758 | Delta GCC: -1.94%
- Clustering medio: 0.3663 | Delta: -33.23%
- Eficiencia global: 0.1345 | Delta: -51.76%
- Caminho medio (gcc): 11.4514 | Delta: +126.14%
- Modularidade: 0.8346 | Delta: +0.1368
- Assortatividade: 0.2368 | Delta: -0.2248
- Comunidades: 20 | Maior comunidade: 23
- Robustez aleatoria (AUC LCC): 0.2498 | Delta: -45.20%
- Robustez alvo (AUC LCC): 0.1543 | Delta: -60.19%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.1676
- Menor correlacao de centralidade entre as metricas: -0.2607
- Overlap medio top-k: 0.2400
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - alterou de forma relevante quais nós aparecem como mais centrais
  - ficou mais frágil a ataques direcionados em hubs

