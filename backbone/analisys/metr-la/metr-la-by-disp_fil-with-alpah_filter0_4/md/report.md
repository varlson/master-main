# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-15 23:27:11

Pasta de saida: `/home/varlson/master/main/master-main/backbone/analisys/metr-la/metr-la-by-disp_fil-with-alpah_filter0_4`

## Dataset: metr-la

### Rede Original

- Nos: 207 | Arestas: 1313 | Densidade: 0.0616
- Grau medio: 12.6860 | Forca media: 5.1448
- Clustering medio: 0.5485 | Eficiencia global: 0.2789 | Modularidade: 0.6978
- Caminho medio (gcc): 5.0637 | Componentes: 2 | Assortatividade: 0.4616

- Comunidades: 8 | Maior comunidade: 53 | Razao do componente gigante: 0.9952

### Comparacoes com a Rede Original

#### metr-la-by-disp_fil-with-alpah_filter0_4

Informacoes basicas do backbone:
- Nos: 207 | Delta: -0.00%
- Arestas: 510 | Delta: -61.16%
- Densidade: 0.0239 | Delta: -61.16%
- Grau medio: 4.9275 | Forca media: 3.4215
- Componentes: 10 | Razao do componente gigante: 0.9130 | Delta GCC: -8.25%
- Clustering medio: 0.4145 | Delta: -24.42%
- Eficiencia global: 0.1504 | Delta: -46.07%
- Caminho medio (gcc): 8.6077 | Delta: +69.99%
- Modularidade: 0.7826 | Delta: +0.0848
- Assortatividade: 0.5633 | Delta: +0.1017
- Comunidades: 21 | Maior comunidade: 33
- Robustez aleatoria (AUC LCC): 0.2981 | Delta: -34.59%
- Robustez alvo (AUC LCC): 0.1789 | Delta: -53.86%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.8754
- Menor correlacao de centralidade entre as metricas: 0.7960
- Overlap medio top-k: 0.7500
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - preservou bem o ranking relativo das centralidades
  - ficou mais frágil a ataques direcionados em hubs

