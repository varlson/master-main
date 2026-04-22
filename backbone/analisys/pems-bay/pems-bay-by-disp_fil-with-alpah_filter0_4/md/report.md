# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-22 14:24:25

Pasta de saida: `/home/suleimane/master/main/master-main/backbone/analisys/pems-bay/pems-bay-by-disp_fil-with-alpah_filter0_4`

## Dataset: pems-bay

### Rede Original

- Nos: 325 | Arestas: 2079 | Densidade: 0.0395
- Grau medio: 12.7938 | Forca media: 6.7824
- Clustering medio: 0.6737 | Eficiencia global: 0.2423 | Modularidade: 0.8198
- Caminho medio (gcc): 5.1616 | Componentes: 7 | Assortatividade: 0.5696

- Comunidades: 19 | Maior comunidade: 67 | Razao do componente gigante: 0.9815

### Comparacoes com a Rede Original

#### pems-bay-by-disp_fil-with-alpah_filter0_4

Informacoes basicas do backbone:
- Nos: 325 | Delta: -0.00%
- Arestas: 1129 | Delta: -45.70%
- Densidade: 0.0214 | Delta: -45.70%
- Grau medio: 6.9477 | Forca media: 5.3546
- Componentes: 10 | Razao do componente gigante: 0.9508 | Delta GCC: -3.13%
- Clustering medio: 0.6679 | Delta: -0.86%
- Eficiencia global: 0.1195 | Delta: -50.69%
- Caminho medio (gcc): 13.9735 | Delta: +170.72%
- Modularidade: 0.8674 | Delta: +0.0475
- Assortatividade: 0.7720 | Delta: +0.2024
- Comunidades: 26 | Maior comunidade: 48
- Robustez aleatoria (AUC LCC): 0.2261 | Delta: -47.66%
- Robustez alvo (AUC LCC): 0.1192 | Delta: -55.65%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.8239
- Menor correlacao de centralidade entre as metricas: 0.7145
- Overlap medio top-k: 0.5600
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - ficou mais frágil a ataques direcionados em hubs

