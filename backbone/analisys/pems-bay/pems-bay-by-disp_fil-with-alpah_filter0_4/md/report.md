# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-15 23:27:38

Pasta de saida: `/home/varlson/master/main/master-main/backbone/analisys/pems-bay/pems-bay-by-disp_fil-with-alpah_filter0_4`

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
- Arestas: 986 | Delta: -52.57%
- Densidade: 0.0187 | Delta: -52.57%
- Grau medio: 6.0677 | Forca media: 4.9247
- Componentes: 18 | Razao do componente gigante: 0.6000 | Delta GCC: -38.87%
- Clustering medio: 0.6585 | Delta: -2.26%
- Eficiencia global: 0.0755 | Delta: -68.85%
- Caminho medio (gcc): 9.4621 | Delta: +83.32%
- Modularidade: 0.8795 | Delta: +0.0597
- Assortatividade: 0.7827 | Delta: +0.2131
- Comunidades: 31 | Maior comunidade: 46
- Robustez aleatoria (AUC LCC): 0.1712 | Delta: -60.36%
- Robustez alvo (AUC LCC): 0.0808 | Delta: -69.91%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.6359
- Menor correlacao de centralidade entre as metricas: 0.1913
- Overlap medio top-k: 0.5900
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - ficou mais frágil a ataques direcionados em hubs

