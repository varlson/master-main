# Resumo Técnico Atualizado (run `20260306_142338`)

Base usada para esta versão:
- `results/old/md/all-datasets_20260306_142338_comparison_report.md`
- `results/old/plots/*_20260306_142338/scatter_real_vs_pred.png`
- `results/old/plots/*_20260306_142338/train_val_curves.png`

## 1) Contexto rápido
Objetivo do projeto: comparar arquiteturas espaço-temporais para previsão de tráfego em grafo (`metr-la` e `pems-bay`) com o mesmo pipeline.

Resultado global do run `20260306_142338`:
- Melhor em `pems-bay`: **STICformer** (`MAE=0.2036`, `RMSE=0.4333`).
- Melhor em `metr-la` por MAE: **DGCRN** (`MAE=0.3077`), mas não é o mais estável visualmente.

## 2) Modelos: em que são baseados e como funcionam

### DCRNN
- **Base:** RNN em grafo com **Diffusion Convolution** (propagação no grafo em caminhadas aleatórias) + célula recorrente.
- **Como funciona:** usa um encoder/decoder temporal e substitui convolução padrão por difusão no grafo, para modelar fluxo de tráfego entre sensores conectados.
- **Ponto forte:** baseline clássico para dependência temporal + topologia.
- **Limitação típica:** pode perder capacidade quando a dinâmica muda muito ou quando há forte heterogeneidade entre nós.

### GraphWaveNet
- **Base:** convoluções temporais dilatadas (estilo WaveNet) + convolução em grafo com matriz de adjacência adaptativa.
- **Como funciona:** extrai padrões temporais em múltiplas escalas com convoluções causais e aprende também conectividades não explícitas no grafo fixo.
- **Ponto forte:** bom equilíbrio entre desempenho, estabilidade e custo de treino.
- **Limitação típica:** pode ter nós específicos com erro alto (outliers) mesmo com média boa.

### MTGNN
- **Base:** aprendizado de estrutura de grafo + blocos temporais/convolucionais + propagação multi-hop.
- **Como funciona:** aprende (ou ajusta) relações entre nós durante o treino e combina isso com modelagem temporal.
- **Ponto forte:** flexível quando a adjacência original não representa toda a dinâmica.
- **Limitação típica:** sensível à regularização e pode abrir gap maior entre treino e validação.

### DGCRN
- **Base:** RNN em grafo com **grafo dinâmico** (a conectividade muda ao longo do tempo/estado).
- **Como funciona:** a célula recorrente usa convolução em grafo onde a estrutura pode variar por passo temporal, tentando capturar não-estacionariedade.
- **Ponto forte:** forte em MAE quando há dinâmica variável.
- **Limitação típica:** risco de sobreajuste alto, principalmente quando o grafo dinâmico fica muito flexível.

### STICformer
- **Base:** Transformer espaço-temporal (atenção multi-cabeça para dependências no tempo e entre nós).
- **Como funciona:** usa atenção para capturar relações de longo alcance temporal e padrões espaciais complexos.
- **Ponto forte:** desempenho robusto e boa generalização no `pems-bay` neste run.
- **Limitação típica:** validação pode oscilar se regularização/early stopping não estiver bem calibrado.

### PatchSTG
- **Base:** Transformer por **patches temporais** (agregação da sequência em blocos) + modelagem espaço-temporal.
- **Como funciona:** em vez de processar cada ponto temporal isoladamente, resume janelas (patches), reduz ruído e custo, e aplica atenção sobre representações mais compactas.
- **Ponto forte:** estabilidade boa e desempenho competitivo em ambos datasets.
- **Limitação típica:** escolha de patch/stride influencia bastante o resultado.

## 3) Comentários por modelo (com base no run `20260306_142338`)

### DCRNN
- **Números:**
  - `metr-la`: `MAE=0.3411`, `RMSE=0.6264`, `MAPE=123.25`.
  - `pems-bay`: `MAE=0.2860`, `RMSE=0.5349`, `MAPE=169.69`.
- **Scatter (`R²`):**
  - `metr-la`: muito heterogêneo (`0.0228` a `0.8618`).
  - `pems-bay`: fraco nos nós críticos (`-2.4675` a `0.6264`).
- **Train/Val:**
  - `metr-la`: train cai bem, val estabiliza ~`0.34` com gap persistente.
  - `pems-bay`: gap alto e pico de validação no meio do treino; sinal claro de overfitting.
- **Leitura:** bom como baseline, mas ficou atrás dos demais neste run.

### GraphWaveNet
- **Números:**
  - `metr-la`: `MAE=0.3256`, `RMSE=0.6207`, `MAPE=113.15`.
  - `pems-bay`: `MAE=0.2152`, `RMSE=0.4528`, `MAPE=121.46`.
- **Scatter (`R²`):**
  - `metr-la`: consistente (`0.7589` a `0.8520`).
  - `pems-bay`: maioria muito boa, com 1 nó crítico baixo (`0.2656` a `0.9591`).
- **Train/Val:**
  - `metr-la`: val oscilante mas controlada, com tendência de queda.
  - `pems-bay`: convergência estável; gap moderado.
- **Leitura:** modelo robusto/estável; não foi o melhor absoluto, mas teve comportamento técnico sólido.

### MTGNN
- **Números:**
  - `metr-la`: `MAE=0.3529`, `RMSE=0.6354`, `MAPE=124.08`.
  - `pems-bay`: `MAE=0.2405`, `RMSE=0.5078`, `MAPE=135.36`.
- **Scatter (`R²`):**
  - `metr-la`: bom (`0.7899` a `0.8957`).
  - `pems-bay`: bom na maioria, 1 nó fraco (`0.3672` a `0.9696`).
- **Train/Val:**
  - `metr-la`: poucas épocas, val com picos; indício de instabilidade.
  - `pems-bay`: val quase lateral e gap relativamente alto no final.
- **Leitura:** potencial espacial existe, mas generalização ficou abaixo dos top models.

### DGCRN
- **Números:**
  - `metr-la`: `MAE=0.3077` (melhor MAE), `RMSE=0.6284`, `MAPE=124.19`.
  - `pems-bay`: `MAE=0.2443`, `RMSE=0.5003`, `MAPE=164.52`.
- **Scatter (`R²`):**
  - `metr-la`: muito bom e estável (`0.8363` a `0.8935`).
  - `pems-bay`: mistura de excelente e fraco (`0.3634` a `0.9658`).
- **Train/Val:**
  - `metr-la` e `pems-bay`: gap extremo (train muito baixo, val alta e oscilante), perfil clássico de overfitting.
- **Leitura:** competitivo em MAE no `metr-la`, mas com risco maior de pouca robustez fora do conjunto de treino.

### STICformer
- **Números:**
  - `metr-la`: `MAE=0.3096`, `RMSE=0.6089`, `MAPE=113.71`.
  - `pems-bay`: `MAE=0.2036` (melhor), `RMSE=0.4333` (melhor), `MAPE=123.63`.
- **Scatter (`R²`):**
  - `metr-la`: forte e estável (`0.8358` a `0.9164`).
  - `pems-bay`: muito forte com 1 nó difícil (`0.4288` a `0.9559`).
- **Train/Val:**
  - `metr-la`: train cai continuamente; val oscila com picos intermediários.
  - `pems-bay`: boa trajetória geral de validação e gap final relativamente controlado.
- **Leitura:** melhor compromisso geral do run entre acurácia e estabilidade.

### PatchSTG
- **Números:**
  - `metr-la`: `MAE=0.3082`, `RMSE=0.6086`, `MAPE=111.69`.
  - `pems-bay`: `MAE=0.2075`, `RMSE=0.4350`, `MAPE=123.13`.
- **Scatter (`R²`):**
  - `metr-la`: estável e alto (`0.8265` a `0.8998`).
  - `pems-bay`: muito bom com 1 nó crítico (`0.4087` a `0.9598`).
- **Train/Val:**
  - `metr-la`: convergência boa e gap moderado.
  - `pems-bay`: val decai e estabiliza em patamar baixo, com oscilações curtas.
- **Leitura:** muito competitivo; ficou imediatamente atrás do STICformer no `pems-bay` e quase empatado no `metr-la`.

## 4) Resumo final
- **Melhor modelo geral deste run:** **STICformer** (melhor em `pems-bay` e entre os mais fortes em `metr-la`).
- **Segundo melhor mais equilibrado:** **PatchSTG** (consistência forte em ambos datasets).
- **Melhor MAE pontual em `metr-la`:** **DGCRN**, mas com evidências visuais de sobreajuste (gap train/val muito alto).
- **Modelo mais estável entre os não-Transformers:** **GraphWaveNet**.
- **Baseline que mais sofreu:** **DCRNN** (principalmente no `pems-bay`, com `R²` muito baixo/negativo em nós difíceis).

Em termos práticos para próximos ciclos: manter **STICformer** e **PatchSTG** como foco principal, usar **GraphWaveNet** como referência de estabilidade e tratar **DGCRN** com regularização/controle de overfitting se for priorizar MAE em `metr-la`.
