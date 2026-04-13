# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-09 00:19:54

Pasta de saida: `/home/varlson/master/main/master-main/backbone/analisys/pems-bay/pems-bay-by-nois_corr-with-alpah_filter0_1`

## Dataset: pems-bay

### Rede Original

- Nos: 325 | Arestas: 2079 | Densidade: 0.0395
- Grau medio: 12.7938 | Forca media: 6.7824
- Clustering medio: 0.6737 | Eficiencia global: 0.2423 | Modularidade: 0.8198
- Caminho medio (gcc): 5.1616 | Componentes: 7 | Assortatividade: 0.5696

- Comunidades: 19 | Maior comunidade: 67 | Razao do componente gigante: 0.9815

### Comparacoes com a Rede Original

#### pems-bay-by-nois_corr-with-alpah_filter0_1

Informacoes basicas do backbone:
- Nos: 325 | Delta: -0.00%
- Arestas: 941 | Delta: -54.74%
- Densidade: 0.0179 | Delta: -54.74%
- Grau medio: 5.7908 | Forca media: 4.0089
- Componentes: 15 | Razao do componente gigante: 0.9385 | Delta GCC: -4.39%
- Clustering medio: 0.5762 | Delta: -14.48%
- Eficiencia global: 0.1385 | Delta: -42.84%
- Caminho medio (gcc): 8.6367 | Delta: +67.33%
- Modularidade: 0.8941 | Delta: +0.0742
- Assortatividade: 0.3359 | Delta: -0.2337
- Comunidades: 31 | Maior comunidade: 34
- Robustez aleatoria (AUC LCC): 0.2903 | Delta: -32.81%
- Robustez alvo (AUC LCC): 0.1912 | Delta: -28.86%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.1981
- Menor correlacao de centralidade entre as metricas: -0.6533
- Overlap medio top-k: 0.1700
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - alterou de forma relevante quais nós aparecem como mais centrais
  - ficou mais frágil a ataques direcionados em hubs

