# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-28 15:05:12

Pasta de saida: `/home/suleimane/master/main/master-main/backbone/analisys/wikivital-mathematics/wikivital-mathematics-by-disp_fil-with-alpah_filter0_3`

## Dataset: wikivital-mathematics

### Rede Original

- Nos: 1068 | Arestas: 27079 | Densidade: 0.0475
- Grau medio: 50.7097 | Forca media: 70.9607
- Clustering medio: 0.4278 | Eficiencia global: 0.4652 | Modularidade: 0.4551
- Caminho medio (gcc): 2.3056 | Componentes: 1 | Assortatividade: -0.0021

- Comunidades: 6 | Maior comunidade: 335 | Razao do componente gigante: 1.0000

### Comparacoes com a Rede Original

#### wikivital-mathematics-by-disp_fil-with-alpah_filter0_3

Informacoes basicas do backbone:
- Nos: 1025 | Delta: -4.03%
- Arestas: 6763 | Delta: -75.02%
- Densidade: 0.0129 | Delta: -72.88%
- Grau medio: 13.1961 | Forca media: 34.0234
- Componentes: 3 | Razao do componente gigante: 0.9961 | Delta GCC: -0.39%
- Clustering medio: 0.3090 | Delta: -27.78%
- Eficiencia global: 0.3166 | Delta: -31.95%
- Caminho medio (gcc): 3.4444 | Delta: +49.39%
- Modularidade: 0.6020 | Delta: +0.1469
- Assortatividade: 0.1000 | Delta: +0.1021
- Comunidades: 12 | Maior comunidade: 243
- Robustez aleatoria (AUC LCC): 0.4632 | Delta: -6.84%
- Robustez alvo (AUC LCC): 0.2722 | Delta: -36.75%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.9414
- Menor correlacao de centralidade entre as metricas: 0.8908
- Overlap medio top-k: 0.8200
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - preservou bem o ranking relativo das centralidades
  - ficou mais frágil a ataques direcionados em hubs

