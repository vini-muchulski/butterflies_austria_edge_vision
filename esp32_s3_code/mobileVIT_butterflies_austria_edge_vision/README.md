# MobileViT Butterflies Austria Edge Vision

## Objetivo

Este diretório contém o estudo de uma MobileViT XX-Small para classificar 20 espécies de borboletas do dataset Butterflies Austria e, posteriormente, executar o modelo quantizado em um ESP32-S3.

O estado atual é experimental, mas a inferência embarcada completa já foi obtida. A exportação foi adaptada ao pipeline do ESP32-S3, `BATCH_MATMUL` foi incorporado ao TFLM e a `FULLY_CONNECTED` per-channel foi corrigida para saídas com rank maior que 2. O trace do PC em modo de referência coincide com o ESP32-S3 até o operador 177. A primeira divergência comprovada aparece na saída da `FULLY_CONNECTED` per-channel do operador 178 e é numericamente pequena. Ainda falta avaliar todo o dataset e remover a instrumentação antes de medir o desempenho final.

## Estrutura atual

```text
mobileVIT_butterflies_austria_edge_vision/
├── README.md
└── esp_idf_mobileVIT_butterflies_austria_edge_vision/
    ├── data/mobilevit_int8.tflite
    ├── python/
    │   ├── inferencia_pc.py
    │   ├── esp_evaluate_inferencia.py
    │   ├── inline_tflite_buffers.py
    │   └── trace_operadores_pc.py
    ├── src/main.cpp
    └── training_code/
        └── butterflies_austria_MobileVIT_apple_baseline_v2_esp_verify/
            ├── best_mobilevit_xx_small_v0.pt
            ├── butterflies_austria_MobileVIT_apple_baseline_v2_esp_verify.ipynb
            └── test1_results/
```

O diretório contém uma aplicação ESP-IDF/PlatformIO para carregar o modelo, expor uma API HTTP e medir memória, hashes e tempo de inferência. Também contém o notebook de treinamento/exportação, os scripts de validação no computador e o avaliador de inferência no ESP32-S3.

## Modelo de origem

O notebook usa `apple/mobilevit-xx-small`, carregado por `MobileViTForImageClassification`, com um classificador para 20 classes. Os pesos treinados estão em `best_mobilevit_xx_small_v0.pt`.

O modelo recebe tensores NCHW:

```text
[1, 3, 256, 256]
```

O pré-processamento de validação e teste é:

1. redimensionar a imagem para 288;
2. aplicar corte central de 256 × 256;
3. converter para RGB;
4. converter os pixels para `[0, 1]`;
5. inverter os canais de RGB para BGR;
6. produzir o tensor no formato CHW.

Não é aplicada normalização ImageNet. O firmware e os scripts de avaliação devem reproduzir exatamente esse processamento.

## Agrupamento estrutural

O notebook cria uma variante agrupada da rede para melhorar a quantização INT8. Canais com escalas de ativação semelhantes são reunidos e determinados blocos residuais invertidos são divididos em ramos.

Configuração usada:

| Bloco | Grupos |
|---:|---:|
| 1 | 8 |
| 2 | 4 |
| 5 | 4 |

Os ramos preservam os pesos da rede original. As saídas são somadas para reconstruir o bloco. A equivalência FP32 medida foi:

| Métrica | Resultado |
|---|---:|
| Concordância top-1 | 99,8457% |
| Erro máximo dos logits | 0,037059 |
| Erro médio dos logits | 0,002080 |
| Similaridade cosseno | 0,999998 |

## Exportação e quantização

O pipeline atual é:

```text
PyTorch → LiteRT FP32 → calibração estratificada → LiteRT INT8
```

A calibração usa 50 imagens por classe do conjunto de treino, totalizando 1.000 imagens. A receita usada é `static_wi8_ai8()`, com pesos e ativações INT8.

Foram exportadas quatro variantes:

| Arquivo | Descrição |
|---|---|
| `mobilevit_original_fp32.tflite` | Arquitetura original em FP32 |
| `mobilevit_original_stratified_int8.tflite` | Arquitetura original quantizada |
| `mobilevit_g8_4_4_fp32.tflite` | Arquitetura agrupada em FP32 |
| `mobilevit_g8_4_4_stratified_int8.tflite` | Arquitetura agrupada quantizada |

O artefato de treinamento candidato é `mobilevit_g8_4_4_stratified_int8.tflite`. O arquivo atualmente usado pelo firmware é `esp_idf_mobileVIT_butterflies_austria_edge_vision/data/mobilevit_int8.tflite`, gerado após as adaptações de exportação descritas abaixo.

## Exportação compatível com ESP32-S3

A célula de exportação do notebook `butterflies_austria_MobileVIT_apple_baseline_v2_esp_verify.ipynb` foi adaptada para produzir um artefato verificável antes do embarque. O fluxo é:

```text
PyTorch treinado → adaptador com AVERAGE_POOL_2D → LiteRT FP32 → buffers inline → quantização INT8 → validação
```

O adaptador substitui a redução espacial final baseada em `SUM` por `F.avg_pool2d` fixo sobre o mapa `8 × 8`, seguido de `flatten` e classificador. A equivalência entre o modelo original e o adaptador é verificada antes da exportação com `torch.testing.assert_close`. Essa alteração não exige novo treinamento, pois preserva os pesos; a quantização deve ser executada novamente para o modelo exportado.

A função genérica de incorporação de buffers lê `Buffer.offset` e `Buffer.size` do FlatBuffer de origem, copia os bytes para `Buffer.data`, zera `offset` e `size` e reempacota o modelo com o identificador `TFL3`. O motivo é que offsets externos são aceitos pelo LiteRT, mas não são resolvidos de forma confiável pelo TFLM usado no firmware. O modelo embarcado deve sempre ser o artefato reempacotado, não o arquivo externo intermediário.

A exportação também valida shape de entrada `[1, 3, 256, 256]`, saída `[1, 20]`, tipo `int8`, versões de operadores e ausência de operadores fora do resolver do firmware. O script independente `python/inline_tflite_buffers.py` mantém a mesma conversão para modelos exportados fora do notebook.

A pré-condição para aceitar um novo modelo é executar, nesta ordem:

1. exportação FP32 com o adaptador compatível;
2. incorporação dos buffers;
3. validação do FlatBuffer e dos operadores;
4. quantização usando calibração representativa;
5. incorporação dos buffers do INT8;
6. validação no LiteRT e comparação de hashes com o firmware;
7. somente depois, cópia para `data/` e gravação no ESP32-S3.

## Resultados no computador

| Modelo | Split | Acurácia | Macro F1 | F1 ponderado |
|---|---|---:|---:|---:|
| Original FP32 | validação | 97,38% | 97,49% | 97,38% |
| Original INT8 | validação | 10,96% | 5,48% | 5,70% |
| Agrupado FP32 | validação | 97,38% | 97,49% | 97,38% |
| Agrupado INT8 | validação | 89,20% | 89,03% | 89,17% |
| Original FP32 | teste | 97,38% | 97,39% | 97,38% |
| Original INT8 | teste | 10,80% | 6,05% | 6,43% |
| Agrupado FP32 | teste | 97,38% | 97,39% | 97,38% |
| Agrupado INT8 | teste | 90,43% | 90,23% | 90,38% |

O agrupamento corrige grande parte da degradação causada pela quantização, mas ainda existe uma perda de aproximadamente 6,94 pontos percentuais em relação ao modelo FP32 no teste.

## Características do candidato INT8 original

Os dados desta seção descrevem o candidato INT8 original, antes da substituição de `SUM` por `AVERAGE_POOL_2D` e da cópia final para `data/`.

```text
Arquivo: mobilevit_g8_4_4_stratified_int8.tflite
Tamanho: 2.071.104 bytes
SHA-256: 4e391c33b40538fec67d98c9ae95590d46544996524e2b3efbdda95aa01cde58
Identificador FlatBuffer: TFL3
Schema: 3
Subgrafos: 1
Operações: 789
Entrada: int8 [1, 3, 256, 256]
Quantização da entrada: scale=0.0038768567610532045, zero_point=-128
Saída: int8 [1, 20]
Quantização da saída: scale=0.031413737684488297, zero_point=-78
```

Operadores presentes:

| Operador | Quantidade |
|---|---:|
| `RESHAPE` | 201 |
| `MUL` | 114 |
| `ADD` | 79 |
| `LOGISTIC` | 60 |
| `TRANSPOSE` | 56 |
| `FULLY_CONNECTED` | 55 |
| `CONV_2D` | 54 |
| `MEAN` | 42 |
| `SQUARED_DIFFERENCE` | 21 |
| `RSQRT` | 21 |
| `SUB` | 21 |
| `DEPTHWISE_CONV_2D` | 20 |
| `BATCH_MATMUL` | 18 |
| `PAD` | 11 |
| `SOFTMAX` | 9 |
| `QUANTIZE` | 3 |
| `CONCATENATION` | 3 |
| `SUM` | 1 |

Também existem versões 1 dos operadores listados e versão 5 de `FULLY_CONNECTED`.

## Estado atual da compatibilidade com ESP32-S3

### Artefato usado no firmware

O firmware atual usa:

```text
Arquivo: esp_idf_mobileVIT_butterflies_austria_edge_vision/data/mobilevit_int8.tflite
Tamanho: 1.843.720 bytes
SHA-256: 1c830eca3f888626e0276d6693270578cc4dbd5d511ee6f5dfd84e87b9c4a149
Entrada: int8 [1, 3, 256, 256]
Entrada: scale=0.0038768567610532045, zero_point=-128
Saída: int8 [1, 20]
```

O arquivo foi carregado pelo LittleFS e o `AllocateTensors()` concluiu no ESP32-S3. Na execução observada, a arena usou `2.881.824` bytes de `5.242.880` bytes.

### Buffers externos

O formato FlatBuffer pode armazenar pesos em `Buffer.offset` e `Buffer.size`. Isso é válido no LiteRT quando o runtime resolve esses offsets, mas não é uma representação segura para o TFLM usado neste projeto.

A conversão implementada copia cada região externa para `Buffer.data`, zera `offset` e `size` e reempacota o modelo. O arquivo final deve ser validado novamente por tamanho, SHA-256, identificador `TFL3`, shapes, quantização e inferência. Não se deve copiar para o firmware o arquivo externo intermediário.

### Redução espacial

O modelo original terminava a extração de características com `SUM` espacial. Essa operação foi substituída no adaptador de exportação por `AVERAGE_POOL_2D` fixo de `8 × 8`, seguido de `flatten` e `FULLY_CONNECTED`. A substituição ocorre no modelo PyTorch antes da conversão, portanto as escalas da quantização são recalculadas corretamente.

Não é suficiente editar o FlatBuffer depois da exportação. Uma alteração estrutural deve ser feita no adaptador, validada contra o modelo original e então exportada e quantizada novamente.

### `BATCH_MATMUL`

A arquitetura MobileViT contém `BATCH_MATMUL` na atenção. Foram adicionados à biblioteca TFLM compartilhada os arquivos de referência:

```text
common_components_butterflies_austria/tflite-lib/tensorflow/lite/micro/kernels/batch_matmul.cc
common_components_butterflies_austria/tflite-lib/tensorflow/lite/micro/kernels/batch_matmul.h
common_components_butterflies_austria/tflite-lib/tensorflow/lite/micro/kernels/batch_matmul_common.cc
common_components_butterflies_austria/tflite-lib/tensorflow/lite/kernels/internal/reference/batch_matmul.h
```

O operador foi registrado no resolver do firmware. A implementação atual é de referência; o ESP-NN permanece sendo usado nos operadores para os quais existe caminho otimizado. A execução embarcada completou os 788 operadores do grafo, incluindo os 18 `BATCH_MATMUL`.

### ESP-NN e quantização

O firmware mantém ESP-NN para `CONV_2D`, `DEPTHWISE_CONV_2D` e `FULLY_CONNECTED` per-tensor. A `FULLY_CONNECTED` foi ajustada para detectar pesos INT8 per-channel, calcular multiplicadores e shifts por canal durante `Prepare()` e usar `reference_integer_ops::FullyConnectedPerChannel()` durante `Eval()`. O caminho ESP-NN continua sendo usado quando a quantização é per-tensor.

A execução completa mantém ESP-NN nos caminhos per-tensor e usa a referência per-channel nas camadas que exigem escalas por canal. Isso preserva as otimizações existentes sem aplicar a rotina per-tensor a pesos que exigem escalas diferentes por canal.

### Correção da `FULLY_CONNECTED` per-channel

O abort no operador 178 foi localizado no seguinte requisito da implementação de referência:

```cpp
TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);
```

Na MobileViT, a primeira `FULLY_CONNECTED` de atenção recebe entrada `[1, 1024, 64]` e produz saída `[1, 1024, 64]`. Os diagnósticos confirmaram índices `485, 90, 91, 486`, tipos `INT8, INT8, INT32, INT8` e ponteiros válidos. Portanto, o problema não era corrupção do nó, ABI, stack ou ESP-NN; era a restrição indevida de saída exclusivamente 2D.

A implementação per-channel foi alinhada à implementação per-tensor: o número de batches passou a ser o produto de todas as dimensões anteriores à última e `output_depth` passou a ser a última dimensão:

```cpp
const int output_dim_count = output_shape.DimensionsCount();
const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
const int output_depth = output_shape.Dims(output_dim_count - 1);
```

Após essa correção, a inferência percorreu os operadores `0` a `787` e terminou sem abort.

### Primeira inferência completa no ESP32-S3

Foi usada a amostra de teste de índice 299, `Lycaenidae/bl-377.jpg`:

| Métrica | Resultado |
|---|---:|
| Classe real | 9 — `Lycaenidae` |
| Classe predita | 9 — `Lycaenidae` |
| Logit máximo | 6,439816 |
| Tempo de inferência instrumentada | 45,675937 s |
| Tempo total no dispositivo | 45,808872 s |
| Arena usada | 2.881.824 bytes |
| Arena alocada | 5.242.880 bytes |
| Modelo | 1.843.720 bytes |
| Memória interna livre após inferência | 249.947 bytes |
| PSRAM livre após inferência | 1.078.600 bytes |

O hash da entrada foi igual no PC e no ESP32-S3: `1b15e6e0`. O hash final foi internamente consistente no dispositivo: `af492174` tanto no último `OPTRACE` quanto na resposta HTTP.

O resultado não foi bit a bit idêntico ao PC. O PC produziu hash `3b61745e`, embora ambos tenham previsto a classe 9. Dezoito dos vinte logits diferiram, com diferença máxima aproximada de `0,25131`, equivalente a oito níveis da saída quantizada. Isso não alterou a classe vencedora, mas exige avaliação no dataset completo.

O tempo de 45,675937 s não é um benchmark final. Ele inclui `OPTRACE`, `MODEL_DIAG`, `NODE_DIAG`, `FC_DIAG`, `FC_TENSOR_DIAG` e `FC_QUANT_DIAG`, além de kernels de referência para `BATCH_MATMUL` e `FULLY_CONNECTED` per-channel.

### Comparação de traces e `IMAGE_INDEX`

A primeira execução de `trace_operadores_pc.py` comparou entradas diferentes:

- ESP32-S3: índice 299, `bl-377.jpg`;
- trace do PC: índice 199, arquivo `tf-018...`.

O motivo é que `inferencia_pc.py` redefine `IMAGE_INDEX = 299`, enquanto `trace_operadores_pc.py` importa `IMAGE_INDEX` de `teste_inferencia.py`, onde o valor era 199. Assim, a diferença já no operador 0 `TRANSPOSE` não representa divergência entre kernels. Antes de comparar traces, deve-se garantir que caminho da imagem e hash de entrada sejam iguais.

A comparação entre `BUILTIN_WITHOUT_DEFAULT_DELEGATES` e `BUILTIN_REF` no PC apresentou a primeira diferença no operador 2 `CONV_2D`, com mesmos extremos e diferença de soma de 27 em 262.144 elementos. Essa diferença pequena é compatível com arredondamentos distintos entre kernels e deve ser separada da comparação entre `BUILTIN_REF` e ESP32-S3.

#### Resolvers usados no PC

`BUILTIN_REF` força os kernels de referência do LiteRT. Esse modo prioriza portabilidade e é a base mais adequada para comparar o comportamento numérico com os kernels de referência do TFLM.

`BUILTIN_WITHOUT_DEFAULT_DELEGATES` desativa delegates automáticos, como XNNPACK, mas ainda usa implementações built-in otimizadas do LiteRT. Portanto, ele não equivale ao modo de referência.

Nenhum dos dois modos usa ESP-NN. ESP-NN é utilizado somente no firmware do ESP32-S3.

### Critérios de compatibilidade

Um novo modelo só pode ser considerado apto quando cumprir todos os itens:

- todos os buffers de pesos incorporados em `Buffer.data`;
- `TFL3` e schema aceitos pelo runtime;
- entrada, saída e tensores intermediários compatíveis com INT8;
- todas as versões de operadores presentes no resolver;
- `AllocateTensors()` concluído com margem de arena;
- inferência completa sem abort ou saída constante;
- hashes e valores intermediários comparados com o computador;
- pré-processamento idêntico ao notebook;
- métricas de acurácia, latência e memória registradas no ESP32-S3.

## Memória

O artefato usado atualmente ocupa 1,76 MiB. O maior tensor individual observado ocupa aproximadamente 1 MiB, mas isso não determina o tamanho da arena. A arena efetivamente usada pelo firmware foi de 2.881.824 bytes.

O consumo deve continuar sendo medido em cada alteração do modelo, mesmo após `AllocateTensors()` concluir no ESP32-S3. A medição deve incluir:

- modelo carregado em PSRAM;
- arena usada e capacidade alocada;
- memória interna livre;
- PSRAM livre;
- buffers temporários de entrada e comunicação;
- pico durante a inferência.

## Por que `strict_full_int8` não comprova compatibilidade

O relatório atual marca o modelo como full INT8 porque entrada e saída são inteiras, não existem operações Flex ou Custom e não existe `DEQUANTIZE` no grafo.

Essa validação comprova compatibilidade com o LiteRT usado no computador. Ela não verifica:

- suporte dos operadores pelo TFLM;
- versões aceitas dos operadores;
- buffers externos;
- diferenças numéricas entre kernels;
- tamanho da tensor arena;
- compatibilidade com ESP-NN;
- layout e pré-processamento implementados no firmware.

## Alterações realizadas no TensorFlow Lite Micro

As alterações funcionais na biblioteca compartilhada foram:

| Arquivo | Alteração |
|---|---|
| `micro/kernels/batch_matmul.cc` | implementação Micro do `BATCH_MATMUL` e execução INT8 de referência |
| `micro/kernels/batch_matmul.h` | estruturas e declarações do operador |
| `micro/kernels/batch_matmul_common.cc` | shapes, transposição e utilitários comuns |
| `kernels/internal/reference/batch_matmul.h` | multiplicação matricial de referência |
| `micro/kernels/micro_ops.h` | declaração de `Register_BATCH_MATMUL()` |
| `micro/micro_mutable_op_resolver.h` | implementação de `AddBatchMatMul()` |
| `micro/all_ops_resolver.cc` | registro do operador no resolver completo |
| `micro/kernels/esp_nn/fully_connected.cc` | preparação e execução per-channel, mantendo ESP-NN no caminho per-tensor |
| `kernels/internal/reference/integer_ops/fully_connected.h` | suporte a saídas com rank maior que 2 |

Os arquivos de `BATCH_MATMUL` vieram da implementação oficial do TensorFlow Lite Micro e foram adaptados apenas ao conjunto de fontes disponível neste repositório. A MobileViT usa 18 instâncias desse operador na atenção.

A implementação original de `FullyConnectedPerChannel()` exigia saída 2D. A MobileViT produz tensores `[1, 1024, 64]`. A correção passou a usar a última dimensão como `output_depth` e o produto das dimensões anteriores como número de batches:

```cpp
const int output_dim_count = output_shape.DimensionsCount();
const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
const int output_depth = output_shape.Dims(output_dim_count - 1);
```

`micro/micro_graph.cc` foi instrumentado temporariamente para acompanhar o nó 178 durante `Prepare()` e `Invoke()`. O diagnóstico confirmou que `builtin_data`, `user_data`, índices, ponteiros e tipos dos tensores permaneciam válidos. Essa instrumentação não faz parte da correção funcional.

O aumento temporário da stack de 16.384 para 32.768 bytes não alterou o abort e foi revertido. Stack insuficiente e corrupção do nó foram descartadas. A causa do abort era exclusivamente a restrição 2D da `FULLY_CONNECTED` per-channel.

## Comparação numérica por operador

A comparação foi repetida com a mesma imagem e o mesmo hash de entrada:

- `BUILTIN_WITHOUT_DEFAULT_DELEGATES` e `BUILTIN_REF` divergem primeiro no operador 2 `CONV_2D`;
- o ESP32-S3 coincide com `BUILTIN_REF` do operador 0 ao operador 177;
- a primeira divergência real entre `BUILTIN_REF` e ESP32-S3 ocorre no operador 178 `FULLY_CONNECTED`;
- ambos produzem `min=-128` e `max=116` no operador 178;
- a soma no PC é `900698` e no ESP32-S3 é `900712`;
- a diferença total é 14 em 65.536 elementos;
- o hash do PC é `3eb38229` e o hash do ESP32-S3 é `7e93fe71`.

Isso mostra que o ESP-NN não é a origem da primeira divergência PC de referência × ESP32-S3. O comprovado é que a primeira diferença aparece na saída da primeira `FULLY_CONNECTED` per-channel da atenção. A requantização é a hipótese principal, mas ainda é necessário comparar acumuladores, multiplicadores, shifts e resultados requantizados elemento a elemento.

O diagnóstico da quantização do operador 178 produziu:

```text
output_depth=64
multiplier_hash=9303cee6
shift_hash=7c816582
multiplier_min=1074935139
multiplier_max=2102005093
shift_min=-9
shift_max=-8
input_offset=21
weights_offset=0
output_offset=13
activation_min=-128
activation_max=127
TFLITE_SINGLE_ROUNDING=0
sizeof(int32_t)=4
sizeof(int)=4
```

Os pesos INT8 são simétricos, portanto `weights_offset=0` é esperado. Tensores, offsets, multiplicadores, shifts e tamanhos dos tipos estão inicializados corretamente. A hipótese de parâmetro não inicializado foi descartada. A diferença restante é compatível com arredondamento ou requantização entre implementações, não com corrupção de memória.

As camadas seguintes executaram normalmente. O classificador final, operador 787, produziu:

```text
tensor=1092
bytes=20
hash=af492174
min=-123
max=127
sum=-1625
```

## Próximos passos

1. localizar os primeiros elementos divergentes no operador 178 e comparar acumulador, multiplicador, shift e resultado requantizado;
2. remover `OPTRACE`, `CONV_DIAG`, `MODEL_DIAG`, `NODE_DIAG`, `FC_DIAG`, `FC_TENSOR_DIAG` e `FC_QUANT_DIAG` após concluir o diagnóstico;
3. medir novamente a latência sem instrumentação;
4. avaliar todo o dataset de teste no dispositivo;
5. registrar acurácia, matriz de confusão, latência, arena, heap interno, PSRAM e hashes para o artigo científico;
6. comparar as métricas completas do ESP32-S3 com `BUILTIN_REF` no PC.

## Artefatos de resultados

Os resultados existentes estão em `test1_results/`:

| Arquivo | Conteúdo |
|---|---|
| `results.csv` | Métricas agregadas das quatro variantes |
| `tflite_inspection.json` | Tipos, shapes, quantização e tamanho |
| `fp32_equivalence.json` | Equivalência entre as arquiteturas FP32 |
| `calibration_manifest.csv` | Imagens usadas na calibração |
| `group_configuration.json` | Canais de cada grupo estrutural |
| `group_statistics.csv` | Estatísticas das escalas por grupo |
| `grouped_operators.csv` | Sequência de operadores do candidato |
| `grouped_int8_validation_report.json` | Métricas por classe na validação |
| `grouped_int8_test_report.json` | Métricas por classe no teste |
| `grouped_int8_validation_confusion_matrix.csv` | Matriz de confusão da validação |
| `grouped_int8_test_confusion_matrix.csv` | Matriz de confusão do teste |
