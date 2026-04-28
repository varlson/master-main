# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-28 14:53:18

Pasta de saida: `/home/suleimane/master/main/master-main/backbone/analisys/metr-la/metr-la-by-glanb-with-alpah_filter0_3`

## Dataset: metr-la

### Rede Original

- Nos: 207 | Arestas: 1313 | Densidade: 0.0616
- Grau medio: 12.6860 | Forca media: 5.1448
- Clustering medio: 0.5485 | Eficiencia global: 0.2789 | Modularidade: 0.6978
- Caminho medio (gcc): 5.0637 | Componentes: 2 | Assortatividade: 0.4616

- Comunidades: 8 | Maior comunidade: 53 | Razao do componente gigante: 0.9952

### Comparacoes com a Rede Original

#### metr-la-by-glanb-with-alpah_filter0_3

Informacoes basicas do backbone:
- Nos: 206 | Delta: -0.48%
- Arestas: 383 | Delta: -70.83%
- Densidade: 0.0181 | Delta: -70.55%
- Grau medio: 3.7184 | Forca media: 2.2355
- Componentes: 1 | Razao do componente gigante: 1.0000 | Delta GCC: +0.49%
- Clustering medio: 0.0990 | Delta: -81.95%
- Eficiencia global: 0.1816 | Delta: -34.86%
- Caminho medio (gcc): 7.8163 | Delta: +54.36%
- Modularidade: 0.7606 | Delta: +0.0628
- Assortatividade: 0.0335 | Delta: -0.4281
- Comunidades: 12 | Maior comunidade: 35
- Robustez aleatoria (AUC LCC): 0.3409 | Delta: -25.20%
- Robustez alvo (AUC LCC): 0.1653 | Delta: -57.36%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.7257
- Menor correlacao de centralidade entre as metricas: 0.5392
- Overlap medio top-k: 0.5300
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - ficou mais frágil a ataques direcionados em hubs

