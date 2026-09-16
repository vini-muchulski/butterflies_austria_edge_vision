# MobileViT Butterflies Austria Edge Vision

## Objetivo

Este diretório contém o estudo de uma MobileViT XX-Small para classificar 20 espécies de borboletas do dataset Butterflies Austria e, posteriormente, executar o modelo quantizado em um ESP32-S3.

O estado atual é experimental. O modelo funciona no LiteRT em computador, mas ainda não está pronto para execução no TensorFlow Lite Micro usado pelo projeto embarcado.

## Estrutura atual

```text
mobileVIT_butterflies_austria_edge_vision/
└── esp_idf_mobileVIT_butterflies_austria_edge_vision/
    └── training_code/
        └── butterflies_austria_MobileVIT_apple_baseline_v1/
            ├── best_mobilevit_xx_small_v0.pt
            ├── butterflies_austria_MobileVIT_apple_baseline_v1.ipynb
            └── test1_results/
```

Ainda não existe uma aplicação ESP-IDF completa neste diretório. O conteúdo atual cobre treinamento, transformação estrutural, exportação, quantização e avaliação no computador.

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

O candidato atual para o ESP32-S3 é `mobilevit_g8_4_4_stratified_int8.tflite`.

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

## Características do candidato INT8

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

## Estado da compatibilidade com ESP32-S3

O modelo ainda não deve ser colocado no firmware como candidato válido. Existem quatro problemas independentes.

### Buffers externos

O arquivo possui 324 buffers. Desses, 78 armazenam os pesos por meio de `Buffer.offset` e `Buffer.size`, em vez de armazená-los em `Buffer.data`.

O LiteRT no computador resolve esses offsets. A versão atual do TensorFlow Lite Micro usada no projeto não os resolve corretamente. Esse é o mesmo formato que causou pesos inválidos no estudo da MobileNetV2.

Antes do embarque, todos os buffers externos devem ser convertidos para `Buffer.data` inline. O arquivo convertido precisa ser validado novamente, incluindo hash, tamanho, inferência e estrutura FlatBuffer.

### `BATCH_MATMUL`

O grafo contém 18 operações `BATCH_MATMUL`, usadas na atenção da MobileViT. A biblioteca TFLM compartilhada atualmente pelo projeto não possui um kernel Micro registrado para esse operador.

Esse é um bloqueio de execução: adicionar o operador ao resolver não é suficiente sem uma implementação de kernel compatível com INT8, shapes e parâmetros usados pelo modelo.

As alternativas são:

1. atualizar o TensorFlow Lite Micro para uma versão que forneça o kernel necessário;
2. portar e validar um kernel Micro de `BATCH_MATMUL`;
3. reescrever a atenção durante a exportação usando operadores já suportados.

### Divergência entre kernels no computador

O modelo foi comparado usando kernels otimizados e kernels de referência do LiteRT. Os kernels otimizados produziram predições variáveis e corretas, enquanto os kernels de referência produziram a mesma saída para imagens diferentes, sempre com classe 8.

A primeira divergência observada aparece na operação 2, uma `CONV_2D`. A diferença inicial é pequena, mas se propaga pelo grafo e termina em colapso da saída. Isso impede assumir que o comportamento do LiteRT otimizado será reproduzido pelo TFLM.

Esse problema deve ser localizado e corrigido antes do teste embarcado. A aprovação exige comparação em múltiplas imagens, ausência de saída constante e métricas próximas entre os dois caminhos.

### Redução final com `SUM`

O fim do grafo atual é:

```text
TRANSPOSE [1, 8, 8, 320] → [1, 320, 8, 8]
SUM sobre os dois eixos espaciais → [1, 320]
FULLY_CONNECTED → [1, 20]
```

Essa redução deve ser substituída na definição exportável por um pooling espacial explícito e fixo, preferencialmente `AVERAGE_POOL_2D` de 8 × 8. O modelo deve ser reexportado e requantizado depois da alteração. Não se deve editar apenas o FlatBuffer, pois a escala da redução faz parte da equivalência numérica.

## Memória

O arquivo ocupa aproximadamente 1,98 MiB. O maior tensor individual observado ocupa aproximadamente 1 MiB, mas isso não determina o tamanho da arena.

O consumo real só poderá ser aprovado depois que o grafo for aceito pelo TFLM e `AllocateTensors()` concluir no ESP32-S3. A medição deve incluir:

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

## Critérios para considerar o modelo embarcável

O modelo somente será considerado apto quando cumprir todos os itens:

- `TFL3` e schema suportado;
- todos os pesos armazenados inline;
- entrada, saída e tensores intermediários compatíveis com INT8;
- nenhum operador sem kernel Micro registrado;
- equivalência aceitável entre LiteRT otimizado e referência;
- pré-processamento do firmware idêntico ao notebook;
- `AllocateTensors()` concluído no ESP32-S3;
- arena e memória com margem operacional;
- inferência correta em um conjunto de imagens conhecido;
- avaliação completa do conjunto de teste no dispositivo;
- acurácia, matriz de confusão, latência e memória registradas.

## Próximos passos

1. alterar a exportação para gerar pooling espacial explícito no lugar de `SUM`;
2. reexportar e requantizar o modelo;
3. converter os buffers externos para inline;
4. validar o arquivo convertido no computador;
5. localizar a propagação da divergência entre kernels otimizados e de referência;
6. definir e validar uma implementação Micro para `BATCH_MATMUL`;
7. criar o firmware MobileViT com resolver explícito;
8. medir arena, memória e latência no ESP32-S3;
9. avaliar o dataset de teste no dispositivo e comparar com os resultados do computador.

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
