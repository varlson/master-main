# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-09 00:19:41

Pasta de saida: `/home/varlson/master/main/master-main/backbone/analisys/pems-bay/pems-bay-by-disp_fil-with-alpah_filter0_1`

## Dataset: pems-bay

### Rede Original

- Nos: 325 | Arestas: 2079 | Densidade: 0.0395
- Grau medio: 12.7938 | Forca media: 6.7824
- Clustering medio: 0.6737 | Eficiencia global: 0.2423 | Modularidade: 0.8198
- Caminho medio (gcc): 5.1616 | Componentes: 7 | Assortatividade: 0.5696

- Comunidades: 19 | Maior comunidade: 67 | Razao do componente gigante: 0.9815

### Comparacoes com a Rede Original

#### pems-bay-by-disp_fil-with-alpah_filter0_1

Informacoes basicas do backbone:
- Nos: 53 | Delta: -83.69%
- Arestas: 7 | Delta: -99.66%
- Densidade: 0.0051 | Delta: -87.14%
- Grau medio: 0.2642 | Forca media: 0.2633
- Componentes: 46 | Razao do componente gigante: 0.0377 | Delta GCC: -96.16%
- Clustering medio: 0.0000 | Delta: -100.00%
- Eficiencia global: 0.0051 | Delta: -97.90%
- Caminho medio (gcc): 1.0000 | Delta: -80.63%
- Modularidade: 0.8571 | Delta: +0.0373
- Assortatividade: nan | Delta: nan
- Comunidades: 46 | Maior comunidade: 2
- Robustez aleatoria (AUC LCC): 0.0309 | Delta: -92.84%
- Robustez alvo (AUC LCC): 0.0208 | Delta: -92.28%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.1588
- Menor correlacao de centralidade entre as metricas: -0.1975
- Overlap medio top-k: 0.0700
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - encurtou os caminhos médios no componente gigante
  - alterou de forma relevante quais nós aparecem como mais centrais
  - ficou mais frágil a ataques direcionados em hubs

