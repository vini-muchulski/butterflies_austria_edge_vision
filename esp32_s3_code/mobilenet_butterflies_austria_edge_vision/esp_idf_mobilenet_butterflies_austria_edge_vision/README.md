# Butterflies Austria MobileNetV2 no ESP32-S3

Classificador MobileNetV2 quantizado em int8 para ESP32-S3, executado com
ESP-IDF, TensorFlow Lite Micro e kernels ESP-NN otimizados. O modelo fica no
LittleFS e recebe pela rede o tensor de entrada já pré-processado e quantizado.

## Modelo

| Item | Valor |
|---|---|
| Modelo | MobileNetV2 |
| Arquivo no projeto | data/model_int8_avgpool.tflite (nome curto: o mklittlefs do PlatformIO limita nomes em data/ a 25 caracteres) |
| Arquivo de origem | training_code/MobileNetV2/butterflies_austria_MobileNetv2/tflite_models_avgpool/model_int8_avgpool.tflite |
| Formato dos pesos | Buffers incorporados em Buffer.data para compatibilidade com o schema do TFLite Micro embarcado |
| Entrada | int8 [1,3,224,224], NCHW |
| Quantização da entrada | scale 0.017977742478251457, zero-point -15 |
| Saída | int8 [1,20] |
| Quantização da saída | scale 0.03021967224776745, zero-point -60 |
| Pós-processamento | logits desquantizados e argmax |
| Operadores | TRANSPOSE, PAD, CONV_2D, DEPTHWISE_CONV_2D, ADD, AVERAGE_POOL_2D, FULLY_CONNECTED |

### Substituição do pooling na exportação

O modelo original realizava o pooling global de uma forma convertida para o
operador `SUM` no TFLite. Na versão INT8, esse operador apresentava forte
divergência numérica entre `BUILTIN_WITHOUT_DEFAULT_DELEGATES` e `BUILTIN_REF`,
portanto o artefato não era adequado para validação no TFLite Micro.

O notebook
`training_code/MobileNetV2/butterflies_austria_MobileNetv2/butterflies_austria_MobileNetv2_avgpool_export.ipynb`
cria `MobileNetV2FixedAvgPool` somente para exportação. O wrapper preserva o
backbone e o classificador e substitui o pooling global por:

~~~python
outputs = F.avg_pool2d(outputs, kernel_size=(7, 7), stride=1)
~~~

Como o feature map de entrada possui dimensão espacial 7x7, essa operação é
equivalente ao pooling global original. A equivalência foi validada contra o
modelo PyTorch original com erro absoluto máximo de `0,0000002384`.

O notebook rejeita qualquer artefato que ainda contenha `SUM` ou que não
contenha `AVERAGE_POOL_2D`. Também valida os modelos FP32, INT8 e INT8A16 e
exige que a diferença de acurácia INT8 entre os resolvers seja de no máximo um
ponto percentual. O modelo usado pelo firmware contém `AVERAGE_POOL_2D` e não
contém `SUM`.

Essa alteração resolveu a divergência causada pelo pooling quantizado. Ela é
independente da conversão posterior de `Buffer.offset` e `Buffer.size` para
`Buffer.data`, necessária para o TFLite Micro embarcado carregar corretamente
os pesos.

Classes na ordem usada pelo ImageFolder do treinamento:

~~~text
0  Aglais_Urticae
1  Anthocharis_Cardamines
2  Apatura_Iris
3  Argynnis_Paphia
4  Colias_Myrmidone
5  Gonepteryx_Rhamni
6  Inachis_Io
7  Iphiclides_Podalirius
8  Lycaena_Virgaureae
9  Lycaenidae
10 Maniola_Jurtina
11 Melanargia_Galathea
12 Nymphalis_Antiopa
13 Papilio_Machaon
14 Parnassius_Apollo
15 Pieris_Rapae
16 Polygonia_C-album
17 Vanessa_Atalanta
18 Vanessa_Cardui
19 Zerynthia_Polyxena
~~~

## Dataset e pré-processamento

O dataset está em:

~~~text
Dataset-butterflies-austria/
├── train/
├── val/
├── test/
└── species.csv
~~~

O caminho padrão dos scripts é Dataset-butterflies-austria três níveis acima
de python/. Use DATASET_ROOT para definir outro caminho.

Pré-processamento de validação e teste:

1. Redimensionar para 224x224.
2. Converter para RGB.
3. Converter para tensor e normalizar com média (0.485, 0.456, 0.406) e desvio
   padrão (0.229, 0.224, 0.225).
4. Reordenar para NCHW.
5. Quantizar para int8 usando a escala e o zero-point da entrada.

O tensor enviado possui 150528 bytes (3 x 224 x 224). O ESP32 não redimensiona
nem normaliza a imagem.

## Pipeline

~~~text
imagem do conjunto de teste
        |
        | resize, RGB, normalização ImageNet, NCHW e quantização int8
        v
int8 [1,3,224,224] --HTTP binário--> ESP32-S3
        |
        | cópia direta para o tensor e MicroInterpreter::Invoke()
        v
int8 [1,20]
        |
        | desquantização dos logits e argmax
        v
classe predita, logits, hashes e telemetria em JSON
~~~

## Carregamento do modelo

1. init_littlefs() monta o LittleFS em /littlefs.
2. load_model() abre kModelPath e mede o arquivo.
3. allocate_memory() tenta PSRAM e usa RAM comum como fallback.
4. O arquivo inteiro é lido para esse buffer.
5. tflite::GetModel() interpreta o FlatBuffer e valida o schema.
6. Uma arena separada é alocada para o TensorFlow Lite Micro.

Ao trocar o arquivo TFLite, grave novamente o filesystem com uploadfs.

### Compatibilidade dos buffers FlatBuffer

O modelo originalmente exportado armazenava 71 buffers de pesos por meio dos
campos Buffer.offset e Buffer.size, com Buffer.data vazio. O schema antigo do
TFLite Micro embarcado reconhece apenas Buffer.data. Esses pesos eram tratados
como tensores sem dados, alocados na arena sem inicialização e sobrescritos
durante a inferência.

O primeiro erro aparecia na operação 5, CONV_2D, tensor de pesos 103:

| Item | Valor |
|---|---:|
| Offset no arquivo original | 2809008 |
| Tamanho | 1536 bytes |
| Hash FNV-1a correto | a6748aa7 |

Antes da conversão, os hashes dos pesos e das ativações divergiam entre
execuções. A rede saturava progressivamente e a FULLY_CONNECTED retornava 20
valores int8 iguais a -128, desquantizados como -1,950634. O resultado incorreto
era sempre a classe 0.

Use inline_tflite_buffers.py para incorporar os buffers antes de gravar o
modelo no LittleFS:

~~~bash
.venv/bin/python python/inline_tflite_buffers.py \
  caminho/modelo_original.tflite \
  data/model_int8_avgpool.tflite
~~~

Origem e destino devem ser arquivos diferentes. Mantenha apenas o modelo
convertido em data/ para não incluir duas cópias no LittleFS.

Modelo convertido validado:

| Item | Valor |
|---|---|
| Tamanho | 2809720 bytes |
| FNV-1a | 30a0d47d |
| SHA-256 | 73444df0dcd482f84e5f923b8178ab6450e8dc308c5af7642594ad3214439dfb |

## Organização

~~~text
esp_idf_mobilenet_butterflies_austria_edge_vision/
├── data/model_int8_avgpool.tflite
├── python/
│   ├── input_info.json
│   ├── get_input_info.py
│   ├── teste_inferencia.py
│   ├── inferencia_pc.py
│   ├── inline_tflite_buffers.py
│   ├── evaluate_inferencia_pc.py
│   ├── sanity_check_host.py
│   └── trace_operadores_pc.py
├── src/main.cpp
├── CMakeLists.txt
├── platformio.ini
├── partitions.csv
└── sdkconfig.defaults
~~~

Os componentes compartilhados ficam em
../../common_components_butterflies_austria. O tflite-lib inclui o ESP-NN e o
firmware registra os kernels necessários no MicroMutableOpResolver.

## Interface HTTP

### GET /status

Retorna estado de inicialização, shape, dtype, tamanho e quantização da
entrada, quantidade de classes, uso da arena e memória livre.

### POST /predict_bin

Requer Content-Type application/octet-stream e exatamente 150528 bytes de
entrada int8 em NCHW. A resposta inclui predicted_class, confidence, scores,
input_fnv1a, output_fnv1a, tempos de execução, uso da arena, tamanho do modelo
e memória livre.

A saída não possui Softmax. confidence é um logit desquantizado, não uma
probabilidade.

## Scripts Python

Execute os scripts a partir de
esp_idf_mobilenet_butterflies_austria_edge_vision.

teste_inferencia.py usa IMAGE_INDEX=299 por padrão, seleciona uma imagem
ordenada do conjunto test, envia o tensor ao ESP32 e salva o resultado e o plot
com a label real e a label predita.

inferencia_pc.py executa a mesma inferência no computador com
BUILTIN_WITHOUT_DEFAULT_DELEGATES e salva JSON, tensor de saída e plot em
results_pc/.

evaluate_inferencia_pc.py executa todas as imagens do conjunto test duas vezes
usando o mesmo arquivo TFLite:

- BUILTIN_WITHOUT_DEFAULT_DELEGATES;
- BUILTIN_REF.

Os resultados são separados em
results_pc/evaluation/builtin_without_default_delegates/ e
results_pc/evaluation/builtin_ref/. Cada diretório contém:

- predictions.csv
- summary.json
- classification_report.txt e classification_report.json
- confusion_matrix.csv e confusion_matrix.png

sanity_check_host.py executa uma ou várias amostras no PC. Sem argumentos, usa
o índice fixo 299.

trace_operadores_pc.py gera tensores intermediários da referência no PC e
compara registros OPTRACE de uma inferência do ESP32.

inline_tflite_buffers.py converte modelos que usam Buffer.offset e Buffer.size
para buffers incorporados em Buffer.data, compatíveis com o TFLite Micro
embarcado.

~~~bash
.venv/bin/python python/get_input_info.py
.venv/bin/python python/inline_tflite_buffers.py modelo_original.tflite data/model_int8_avgpool.tflite
.venv/bin/python python/inferencia_pc.py
.venv/bin/python python/evaluate_inferencia_pc.py
.venv/bin/python python/sanity_check_host.py 299
.venv/bin/python python/trace_operadores_pc.py caminho/do/log_serial.txt
ESP32_IP=192.168.3.22 .venv/bin/python python/teste_inferencia.py
~~~

Variáveis aceitas incluem DATASET_ROOT, MODEL_PATH, ESP32_IP, INPUT_INFO_PATH
e OUTPUT_PLOT_PATH.

## Validação PC e ESP32

1. Execute evaluate_inferencia_pc.py para validar o modelo inteiro nos dois
   resolvers do PC.
2. Execute inferencia_pc.py para gerar a referência da amostra.
3. Envie a mesma amostra com teste_inferencia.py.
4. Compare primeiro input_fnv1a.
5. Depois compare classe, logits e output_fnv1a.
6. Use trace_operadores_pc.py somente para localizar divergências.

Resultados validados no conjunto test com 648 imagens:

| Resolver | Acurácia |
|---|---:|
| BUILTIN_WITHOUT_DEFAULT_DELEGATES | 97,38% |
| BUILTIN_REF | 97,22% |

O modelo INT8 com AVERAGE_POOL_2D produz resultados equivalentes nos dois resolvers. Na imagem 299, ambos predizem Lycaenidae (classe 9).

A entrada da imagem 299 possui hash f84ff650. O firmware deve usar os mesmos parâmetros de quantização registrados em python/input_info.json.

Com o modelo convertido e os kernels de referência, as operações 0 a 68 do
ESP32 coincidem exatamente com o resolver BUILTIN_REF do PC para a imagem 299.
A FULLY_CONNECTED foi corrigida para usar multiplicadores e shifts per-channel;
depois dessa correção, a operação 69 e a saída final também coincidem.

Com ESP-NN otimizado, alguns tensores intermediários possuem hashes diferentes
da referência por diferenças de arredondamento dos kernels. Algumas operações
posteriores voltam a coincidir e a saída final permanece exatamente igual. Para
a imagem 299, os hashes validados são:

| Tensor | FNV-1a |
|---|---|
| Entrada | f84ff650 |
| Saída | 0cad259a |

O resultado final é Lycaenidae, classe 9, com logit 4,968021.

Tempos medidos no ESP32-S3 a 240 MHz, mantendo OPTRACE, CONV_DIAG e MODEL_DIAG
ativos:

| Implementação | Tempo de inferência | Ganho sobre referência |
|---|---:|---:|
| TFLite Micro de referência | 135,861708 s | 1,00x |
| ESP-NN ANSI C | 19,169479 s | 7,09x |
| ESP-NN otimizado para ESP32-S3 | 3,288662 s | 41,31x |

O backend otimizado é 5,83x mais rápido que o backend ESP-NN ANSI C. Esses
tempos incluem a instrumentação de diagnóstico e não representam o desempenho
final sem tracing.

### Avaliação completa no ESP32-S3

A avaliação concluída em 16 de setembro de 2026 executou as 648 imagens do
conjunto de teste no ESP32-S3, após três inferências de aquecimento:

| Métrica | Resultado |
|---|---:|
| Amostras | 648 |
| Acertos | 630 |
| Erros | 18 |
| Acurácia | 97,2222% |
| IC 95% de Wilson | 95,6520%–98,2358% |
| Macro precision | 0,9731 |
| Macro recall | 0,9705 |
| Macro F1 | 0,9714 |
| Precision ponderada | 0,9732 |
| Recall ponderado | 0,9722 |
| F1 ponderado | 0,9723 |

Tempos agregados:

| Métrica | Média | Mediana | p95 | Mínimo–máximo |
|---|---:|---:|---:|---:|
| Inferência no ESP32-S3 | 3,269903 s | 3,269282 s | 3,274134 s | 3,265131–3,283595 s |
| Processamento total no dispositivo | 3,456954 s | 3,456316 s | 3,461416 s | 3,452882–3,470506 s |
| Round-trip HTTP | 5,037682 s | 4,544239 s | 7,813995 s | 3,685531–9,301375 s |
| Recepção no dispositivo | 1,378642 s | 0,898931 s | 4,000345 s | 0,171196–5,323385 s |
| Cópia da entrada | 16,064 ms | 16,005 ms | 16,545 ms | 15,920–16,964 ms |

| Métrica operacional | Resultado |
|---|---:|
| Duração da avaliação | 55 min 7,765 s |
| Duração total do run | 55 min 25,712 s |
| Throughput | 0,195903 imagem/s |
| Modelo | 2.809.720 bytes |
| Arena usada | 2.599.136 bytes |
| Arena alocada | 5.242.880 bytes |
| Uso da arena | 49,57% |
| Margem da arena | 2.643.744 bytes |
| Memória interna livre média após inferência | 250.607 bytes |
| Memória interna livre mínima após inferência | 250.519 bytes |
| PSRAM livre após inferência | 173.384 bytes |

O round-trip inclui envio do tensor por Wi-Fi, espera pelo dispositivo e
recepção da resposta. A métrica de inferência é a medida interna do
`MicroInterpreter::Invoke()`. Essa versão do avaliador registrou memória
somente após cada inferência; não há valores anteriores à chamada nem mínimos
históricos durante a execução.

O artefato avaliado foi:

~~~text
data/model_int8_avgpool.tflite
Tamanho: 2.809.720 bytes
FNV-1a: 30a0d47d
SHA-256: 73444df0dcd482f84e5f923b8178ab6450e8dc308c5af7642594ad3214439dfb
~~~

Os resultados completos estão em
`python/results_esp/evaluation/20260916T142132Z_model_int8_avgpool/`, incluindo
`predictions.csv`, `summary.json`, relatório de classificação, matriz de
confusão, telemetria e log do terminal.

## Memória e concorrência

A configuração foi preparada para ESP32-S3 N16R8:

- CPU a 240 MHz;
- flash QIO de 16 MB a 80 MHz;
- PSRAM octal de 8 MB a 80 MHz;
- partição de aplicação de 6 MB;
- partição LittleFS de aproximadamente 9,88 MB;
- arena de 5 MiB em PSRAM (aproximadamente 2,60 MB usados com o modelo
  convertido; 2.599.136 bytes na configuração ESP-NN otimizada validada).

A requisição HTTP executa de forma síncrona e existe uma única instância do
interpretador. O firmware pressupõe uma inferência por vez.

## Compilação e gravação

O tflite-lib está compilado com -DESP_NN. O sdkconfig.esp32s3 seleciona
CONFIG_NN_OPTIMIZED=y e CONFIG_NN_OPTIMIZATIONS=1. A FULLY_CONNECTED com pesos
per-channel usa o kernel inteiro de referência corrigido. O componente inclui
tensorflow/lite/kernels/internal/common.cc. A flag temporária
-DTFLITE_SINGLE_ROUNDING foi removida. Para evitar o erro de múltiplas fontes
gerando kernel_util.cc.o, somente tensorflow/lite/kernels/kernel_util.cc é
compilado; tensorflow/lite/micro/kernels/kernel_util.cc permanece fora da lista
de fontes.

A compilação e a gravação ficam a cargo do desenvolvedor:

~~~bash
pio run -t clean
pio run
pio run -t uploadfs
pio run -t upload -t monitor
~~~

Use uploadfs na primeira gravação e sempre que data/ for alterado. Uma alteração
somente no C++ exige apenas gravar o firmware.

## Estado atual

- O modelo e os scripts estão configurados para Butterflies Austria e
  MobileNetV2 int8.
- O modelo em data/ usa buffers incorporados e possui FNV-1a 30a0d47d.
- O modelo INT8 com AVERAGE_POOL_2D foi validado com BUILTIN_WITHOUT_DEFAULT_DELEGATES e BUILTIN_REF.
- A avaliação completa no ESP32-S3 obteve 97,2222% de acurácia em 648 imagens.
- A imagem 299 foi validada no ESP32 como Lycaenidae, classe 9.
- A configuração atual usa ESP-NN otimizado e FULLY_CONNECTED per-channel de
  referência.
- A saída otimizada foi validada com o mesmo hash da referência: 0cad259a.
- OPTRACE deve ser removido ou desabilitado antes de medir o desempenho final.
- SSID e senha ainda estão em src/main.cpp; mova-os para configuração local ou
  NVS antes de distribuir o firmware.
- O mklittlefs do PlatformIO rejeita nomes de arquivos em data/ com mais de 25
  caracteres; mantenha o modelo com um nome curto (model_int8_avgpool.tflite) e
  alinhe kModelPath em src/main.cpp.
