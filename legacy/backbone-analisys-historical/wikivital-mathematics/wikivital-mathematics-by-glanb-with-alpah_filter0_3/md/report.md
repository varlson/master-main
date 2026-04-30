# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-28 15:16:20

Pasta de saida: `/home/suleimane/master/main/master-main/backbone/analisys/wikivital-mathematics/wikivital-mathematics-by-glanb-with-alpah_filter0_3`

## Dataset: wikivital-mathematics

### Rede Original

- Nos: 1068 | Arestas: 27079 | Densidade: 0.0475
- Grau medio: 50.7097 | Forca media: 70.9607
- Clustering medio: 0.4278 | Eficiencia global: 0.4652 | Modularidade: 0.4551
- Caminho medio (gcc): 2.3056 | Componentes: 1 | Assortatividade: -0.0021

- Comunidades: 6 | Maior comunidade: 335 | Razao do componente gigante: 1.0000

### Comparacoes com a Rede Original

#### wikivital-mathematics-by-glanb-with-alpah_filter0_3

Informacoes basicas do backbone:
- Nos: 1065 | Delta: -0.28%
- Arestas: 4745 | Delta: -82.48%
- Densidade: 0.0084 | Delta: -82.38%
- Grau medio: 8.9108 | Forca media: 23.8178
- Componentes: 1 | Razao do componente gigante: 1.0000 | Delta GCC: 0.00%
- Clustering medio: 0.2301 | Delta: -46.23%
- Eficiencia global: 0.3248 | Delta: -30.20%
- Caminho medio (gcc): 3.3066 | Delta: +43.42%
- Modularidade: 0.5497 | Delta: +0.0946
- Assortatividade: -0.1567 | Delta: -0.1546
- Comunidades: 11 | Maior comunidade: 217
- Robustez aleatoria (AUC LCC): 0.4575 | Delta: -7.99%
- Robustez alvo (AUC LCC): 0.1604 | Delta: -62.73%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.9311
- Menor correlacao de centralidade entre as metricas: 0.8535
- Overlap medio top-k: 0.8700
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - preservou bem o ranking relativo das centralidades
  - ficou mais frágil a ataques direcionados em hubs

