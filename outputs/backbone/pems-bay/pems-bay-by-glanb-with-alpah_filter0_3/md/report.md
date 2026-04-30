# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-29 21:09:11

Pasta de saida: `/home/suleimane/master/main/master-main/outputs/backbone/pems-bay/pems-bay-by-glanb-with-alpah_filter0_3`

## Dataset: pems-bay

### Rede Original

- Nos: 325 | Arestas: 2079 | Densidade: 0.0395
- Grau medio: 12.7938 | Forca media: 6.7824
- Clustering medio: 0.6737 | Eficiencia global: 0.2423 | Modularidade: 0.8198
- Caminho medio (gcc): 5.1616 | Componentes: 7 | Assortatividade: 0.5696

- Comunidades: 19 | Maior comunidade: 67 | Razao do componente gigante: 0.9815

### Comparacoes com a Rede Original

#### pems-bay-by-glanb-with-alpah_filter0_3

Informacoes basicas do backbone:
- Nos: 319 | Delta: -1.85%
- Arestas: 622 | Delta: -70.08%
- Densidade: 0.0123 | Delta: -68.94%
- Grau medio: 3.8997 | Forca media: 2.4914
- Componentes: 1 | Razao do componente gigante: 1.0000 | Delta GCC: +1.88%
- Clustering medio: 0.1668 | Delta: -75.25%
- Eficiencia global: 0.1746 | Delta: -27.94%
- Caminho medio (gcc): 7.3573 | Delta: +42.54%
- Modularidade: 0.8235 | Delta: +0.0036
- Assortatividade: 0.0077 | Delta: -0.5619
- Comunidades: 13 | Maior comunidade: 50
- Robustez aleatoria (AUC LCC): 0.3318 | Delta: -23.19%
- Robustez alvo (AUC LCC): 0.0916 | Delta: -65.90%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.7767
- Menor correlacao de centralidade entre as metricas: 0.4684
- Overlap medio top-k: 0.5700
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - ficou mais frágil a ataques direcionados em hubs

