# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-24 21:25:25

Pasta de saida: `/home/suleimane/master/main/master-main/backbone/analisys/pems-bay/pems-bay-by-disp_fil-with-alpah_filter0_3`

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
- Arestas: 858 | Delta: -58.73%
- Densidade: 0.0163 | Delta: -58.73%
- Grau medio: 5.2800 | Forca media: 4.4338
- Componentes: 21 | Razao do componente gigante: 0.4215 | Delta GCC: -57.05%
- Clustering medio: 0.6493 | Delta: -3.62%
- Eficiencia global: 0.0536 | Delta: -77.87%
- Caminho medio (gcc): 7.7657 | Delta: +50.45%
- Modularidade: 0.8977 | Delta: +0.0778
- Assortatividade: 0.7447 | Delta: +0.1751
- Comunidades: 32 | Maior comunidade: 42
- Robustez aleatoria (AUC LCC): 0.1376 | Delta: -68.15%
- Robustez alvo (AUC LCC): 0.0867 | Delta: -67.74%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.7414
- Menor correlacao de centralidade entre as metricas: 0.5036
- Overlap medio top-k: 0.5500
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - ficou mais frágil a ataques direcionados em hubs

