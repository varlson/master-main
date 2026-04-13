# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-09 00:21:14

Pasta de saida: `/home/varlson/master/main/master-main/backbone/analisys/pems-bay/pems-bay-by-disp_fil-with-alpah_filter0_3`

## Dataset: pems-bay

### Rede Original

- Nos: 325 | Arestas: 2079 | Densidade: 0.0395
- Grau medio: 12.7938 | Forca media: 6.7824
- Clustering medio: 0.6737 | Eficiencia global: 0.2423 | Modularidade: 0.8198
- Caminho medio (gcc): 5.1616 | Componentes: 7 | Assortatividade: 0.5696

- Comunidades: 19 | Maior comunidade: 67 | Razao do componente gigante: 0.9815

### Comparacoes com a Rede Original

#### pems-bay-by-disp_fil-with-alpah_filter0_3

Informacoes basicas do backbone:
- Nos: 325 | Delta: -0.00%
- Arestas: 695 | Delta: -66.57%
- Densidade: 0.0132 | Delta: -66.57%
- Grau medio: 4.2769 | Forca media: 3.7745
- Componentes: 32 | Razao do componente gigante: 0.3323 | Delta GCC: -66.14%
- Clustering medio: 0.6035 | Delta: -10.42%
- Eficiencia global: 0.0379 | Delta: -84.38%
- Caminho medio (gcc): 8.0817 | Delta: +56.57%
- Modularidade: 0.9061 | Delta: +0.0863
- Assortatividade: 0.7561 | Delta: +0.1865
- Comunidades: 42 | Maior comunidade: 35
- Robustez aleatoria (AUC LCC): 0.1069 | Delta: -75.26%
- Robustez alvo (AUC LCC): 0.0532 | Delta: -80.19%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.4809
- Menor correlacao de centralidade entre as metricas: 0.0476
- Overlap medio top-k: 0.5100
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - alterou de forma relevante quais nós aparecem como mais centrais
  - ficou mais frágil a ataques direcionados em hubs

