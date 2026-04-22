# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-22 14:24:05

Pasta de saida: `/home/suleimane/master/main/master-main/backbone/analisys/metr-la/metr-la-by-disp_fil-with-alpah_filter0_4`

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
- Arestas: 585 | Delta: -55.45%
- Densidade: 0.0274 | Delta: -55.45%
- Grau medio: 5.6522 | Forca media: 3.7094
- Componentes: 6 | Razao do componente gigante: 0.9420 | Delta GCC: -5.34%
- Clustering medio: 0.4130 | Delta: -24.71%
- Eficiencia global: 0.1722 | Delta: -38.25%
- Caminho medio (gcc): 7.8618 | Delta: +55.26%
- Modularidade: 0.7742 | Delta: +0.0764
- Assortatividade: 0.5677 | Delta: +0.1061
- Comunidades: 14 | Maior comunidade: 34
- Robustez aleatoria (AUC LCC): 0.3372 | Delta: -26.02%
- Robustez alvo (AUC LCC): 0.2330 | Delta: -39.91%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.9310
- Menor correlacao de centralidade entre as metricas: 0.8928
- Overlap medio top-k: 0.8100
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - preservou bem o ranking relativo das centralidades
  - ficou mais frágil a ataques direcionados em hubs

