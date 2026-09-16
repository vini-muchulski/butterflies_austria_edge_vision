# Plano de migração: CIFAR-10 para Butterflies Austria MobileNetV2

## Objetivo

Embarcar `training_code/MobileNetV2/butterflies_austria_MobileNetv2/tflite_models/model_int8.tflite` no ESP32-S3 usando TensorFlow Lite Micro e ESP-NN, preservando no cliente o redimensionamento, a normalização, a quantização e a organização do tensor usados no treinamento.

## Contrato do modelo de destino

| Item | Valor |
|---|---|
| Arquivo de origem | `training_code/MobileNetV2/butterflies_austria_MobileNetv2/tflite_models/model_int8.tflite` |
| Tamanho | `2.813.232 bytes` |
| Entrada | `int8 [1,3,224,224]` |
| Layout | NCHW |
| Scale da entrada | `0.017977742478251457` |
| Zero-point da entrada | `-15` |
| Saída | `int8 [1,20]` |
| Scale da saída | `0.030340557917952538` |
| Zero-point da saída | `-58` |
| Pós-processamento | Desquantização e `argmax` sobre logits |
| Operadores | `TRANSPOSE`, `PAD`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `SUM`, `FULLY_CONNECTED` |

## Ordem das classes

Não alterar esta ordem, pois ela é definida pelo `ImageFolder` usado no treinamento:

| Índice | Classe |
|---:|---|
| 0 | `Aglais_Urticae` |
| 1 | `Anthocharis_Cardamines` |
| 2 | `Apatura_Iris` |
| 3 | `Argynnis_Paphia` |
| 4 | `Colias_Myrmidone` |
| 5 | `Gonepteryx_Rhamni` |
| 6 | `Inachis_Io` |
| 7 | `Iphiclides_Podalirius` |
| 8 | `Lycaena_Virgaureae` |
| 9 | `Lycaenidae` |
| 10 | `Maniola_Jurtina` |
| 11 | `Melanargia_Galathea` |
| 12 | `Nymphalis_Antiopa` |
| 13 | `Papilio_Machaon` |
| 14 | `Parnassius_Apollo` |
| 15 | `Pieris_Rapae` |
| 16 | `Polygonia_C-album` |
| 17 | `Vanessa_Atalanta` |
| 18 | `Vanessa_Cardui` |
| 19 | `Zerynthia_Polyxena` |

## 1. Corrigir a dependência de componentes

Arquivo: `CMakeLists.txt`

Substituir:

```cmake
"${CMAKE_CURRENT_LIST_DIR}/../common_components_Cifar10_MobileNetv2"
```

Por:

```cmake
"${CMAKE_CURRENT_LIST_DIR}/../../common_components_butterflies_austria"
```

Verificação:

- O diretório deve conter `tflite-lib` e `esp-nn`.
- Uma configuração limpa não pode depender dos caminhos absolutos armazenados em `.pio/build`.

## 2. Instalar o modelo no LittleFS

Copiar:

```text
training_code/MobileNetV2/butterflies_austria_MobileNetv2/tflite_models/model_int8.tflite
```

Para:

```text
data/mobilenetv2_int8.tflite
```

Verificação:

- Confirmar que os dois arquivos possuem o mesmo SHA-256.
- Usar nome de até 25 caracteres: o mklittlefs do PlatformIO rejeita nomes
  maiores em `data/`.
- Confirmar que o modelo cabe na partição LittleFS de `0x9E0000` bytes.
- Gravar novamente o filesystem quando o arquivo em `data/` mudar.

## 3. Atualizar identificação e dimensões no firmware

Arquivo: `src/main.cpp`

Alterar:

```cpp
constexpr const char *kModelName = "Butterflies Austria MobileNetV2";
constexpr const char *kModelPath = "/littlefs/mobilenetv2_int8.tflite";
```

Manter:

```cpp
static constexpr int kInputWidth = 224;
static constexpr int kInputHeight = 224;
static constexpr int kInputChannels = 3;
static constexpr int kImageSize = kInputWidth * kInputHeight * kInputChannels;
```

Alterar:

```cpp
static constexpr int kOutputClasses = 20;
```

Verificação:

- `kImageSize` deve permanecer `150528` bytes.
- O tamanho permanece igual porque `uint8` e `int8` ocupam um byte, apesar da mudança de tipo e layout.

## 4. Atualizar o resolver de operadores

Arquivo: `src/main.cpp`

Usar `MicroMutableOpResolver<7>` com somente:

```cpp
AddConv2D()
AddDepthwiseConv2D()
AddAdd()
AddFullyConnected()
AddPad()
AddSum()
AddTranspose()
```

Remover:

```cpp
AddQuantize()
AddMean()
AddSoftmax()
```

Verificação:

- `AllocateTensors()` não deve reportar operador ausente.
- Não adicionar operadores que não estejam no modelo.

## 5. Atualizar a validação dos tensores

Arquivo: `src/main.cpp`

Validar a entrada como:

```text
tipo: int8
shape: [1,3,224,224]
bytes: 150528
scale: valor finito e positivo
```

Interpretar as dimensões como:

```text
dims[0] = batch
dims[1] = canais
dims[2] = altura
dims[3] = largura
```

Validar a saída como:

```text
tipo: int8
shape: [1,20]
bytes: 20
scale: valor finito e positivo
```

Remover mensagens específicas de CIFAR-10.

Verificação:

- Registrar no boot o tipo, shape, scale e zero-point reais.
- Interromper a inicialização se qualquer campo divergir.

## 6. Atualizar a cópia da entrada

Arquivo: `src/main.cpp`

Alterar o contrato semântico de entrada de `uint8 NHWC` para `int8 NCHW`.

Copiar os bytes recebidos para:

```cpp
model_ctx.input_tensor->data.int8
```

O buffer HTTP pode continuar sendo alocado como bytes sem sinal, desde que nenhuma conversão numérica seja aplicada antes do `memcpy`.

Verificação:

- O hash FNV-1a deve ser calculado sobre os bytes exatos enviados pelo cliente.
- O hash da entrada no PC e no ESP32 deve coincidir.

## 7. Atualizar o pós-processamento

Arquivo: `src/main.cpp`

Manter a desquantização:

```text
score = (valor_int8 - zero_point) * scale
```

Executar `argmax` sobre os 20 scores.

Não tratar os scores como probabilidades. O modelo retorna logits porque não contém `Softmax`.

Verificação:

- A classe prevista deve ser o índice do maior logit.
- `scores` deve conter exatamente 20 valores.
- `confidence` deve ser documentado como logit desquantizado ou renomeado para `score`.

## 8. Atualizar respostas HTTP e textos fixos

Arquivo: `src/main.cpp`

Atualizar:

- `/status`: tipo `int8`, shape `[1,3,224,224]`, layout `NCHW` e 20 classes.
- `/`: remover referências a CIFAR-10 e `uint8 NHWC`.
- `/predict_bin`: manter `Content-Length` igual a `150528`.
- Mensagens de erro: remover referências a CIFAR-10.
- Resposta de inferência: emitir 20 scores.

Verificação:

- `GET /status` deve refletir o tensor obtido do interpretador, sem metadados antigos hardcoded.

## 9. Atualizar o gerador de metadados

Arquivo: `python/get_input_info.py`

Alterar `MODEL_PATH` para o modelo em `data/`.

Gravar no JSON:

```text
shape
layout
dtype
scale
zero_point
```

Resultado esperado:

```json
{
  "shape": [1, 3, 224, 224],
  "layout": "NCHW",
  "dtype": "int8",
  "scale": 0.017977742478251457,
  "zero_point": -15
}
```

Verificação:

- O script deve falhar se a entrada não for `int8 [1,3,224,224]`.

## 10. Substituir o cliente CIFAR-10

Arquivo: `python/teste_inferencia.py`

Remover:

- Carregamento do dataset CIFAR-10.
- Lista de classes CIFAR-10.
- Normalização `pixel / 127.5 - 1.0`.
- Produção de tensor `uint8 NHWC`.

Implementar a leitura de uma imagem RGB e reproduzir o pipeline de validação do treinamento:

```text
1. Converter para RGB.
2. Redimensionar para 224x224 com interpolação bilinear.
3. Converter pixels para float32 no intervalo [0,1].
4. Normalizar cada canal com mean e std do ImageNet.
5. Converter HWC para CHW.
6. Quantizar usando scale e zero-point da entrada.
7. Aplicar clip em [-128,127].
8. Converter para int8.
9. Adicionar batch, produzindo [1,3,224,224].
10. Enviar os 150528 bytes sem transformação adicional.
```

Constantes:

```python
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
```

Quantização:

```text
q = round(normalizado / scale + zero_point)
q = clip(q, -128, 127)
```

Verificação:

- Shape final: `[1,3,224,224]`.
- Tipo final: `int8`.
- Quantização deve usar os metadados lidos do modelo, não valores duplicados no cliente.

## 11. Atualizar a referência no PC

Arquivo: `python/inferencia_pc.py`

Alterar:

- Caminho do modelo.
- Validação da entrada para `int8 [1,3,224,224]`.
- Validação da saída para `int8 [1,20]`.
- Dataset CIFAR-10 por imagem fornecida pelo usuário.
- Lista de classes para as 20 classes de borboletas.
- Pré-processamento para o pipeline definido na etapa 10.
- Nomes de arquivos e diretórios de resultado.

Verificação:

- Executar o modelo com `BUILTIN_REF`.
- Salvar a entrada INT8, saída INT8, hashes, logits desquantizados e classe prevista.

## 12. Atualizar scripts auxiliares

Arquivos:

- `python/sanity_check_host.py`
- `python/trace_operadores_pc.py`

Alterar referências a CIFAR-10, nomes de resultados, carregamento de imagem e quantidade de classes.

O trace esperado possui 70 operadores reais. O operador `DELEGATE` exibido pelo interpretador desktop não pertence ao FlatBuffer e não deve ser registrado no TFLM.

Verificação:

- O trace do PC deve usar `BUILTIN_REF` e preservar todos os tensores.
- Comparar primeiro o hash da entrada e depois a primeira divergência de operador.

## 13. Validar memória na placa

Configurar:

```cpp
static constexpr int kTensorArenaSize = 5 * 1024 * 1024;
```

Após `AllocateTensors()`:

- Registrar `arena_used_bytes()`.
- Confirmar que modelo, arena, buffer HTTP, Wi-Fi e servidor cabem simultaneamente na PSRAM.
- Aumentar a arena somente se `AllocateTensors()` falhar por memória.
- Reduzir a arena somente após medir o uso real com margem operacional.

Valores medidos no boot:

```text
modelo: 2.813.232 bytes (aproximadamente 2,68 MiB)
arena configurada: 5.242.880 bytes (5 MiB)
arena usada: 4.865.808 bytes (folga de aproximadamente 371 KiB)
entrada HTTP: 150.528 bytes
PSRAM consumida na inicialização: 8.060.936 bytes
```

## 14. Validar equivalência PC e ESP32

Executar a validação nesta ordem:

1. Confirmar contrato no `GET /status`.
2. Preparar uma imagem no cliente.
3. Salvar os bytes INT8 enviados.
4. Executar os mesmos bytes no interpretador de referência do PC.
5. Enviar os bytes ao ESP32.
6. Confirmar igualdade do hash FNV-1a da entrada.
7. Comparar classe prevista.
8. Comparar os 20 valores INT8 de saída.
9. Comparar hash da saída.
10. Se houver divergência, usar o trace por operador.

Critério mínimo de aceite:

- Mesmo hash de entrada.
- Mesmo índice previsto.
- Nenhum erro em `AllocateTensors()` ou `Invoke()`.
- Memória livre estável entre inferências repetidas.

Critério de equivalência exata:

- Mesmo hash de saída INT8 entre `BUILTIN_REF` e ESP32.

Diferenças de arredondamento dos kernels ESP-NN podem impedir igualdade byte a byte. Nesse caso, registrar a primeira divergência, o erro máximo dos logits e confirmar estabilidade da classe.

## 15. Atualizar documentação

Arquivos:

- `README.md`
- `comandos.txt`

Remover referências a:

- CIFAR-10.
- `uint8 NHWC`.
- 10 classes.
- `Softmax`.
- Diretórios de outros projetos.

Documentar:

- Modelo Butterflies Austria MobileNetV2.
- Entrada `int8 NCHW`.
- Pré-processamento ImageNet.
- 20 classes na ordem correta.
- Saída em logits.
- Necessidade de gravar LittleFS após trocar o modelo.

## 16. Corrigir credenciais

Arquivo: `src/main.cpp`

Remover SSID e senha diretamente do código-fonte. Carregar as credenciais por configuração local não versionada ou por NVS.

Verificação:

- Nenhuma credencial real deve permanecer no código ou na documentação.

## Checklist final

- [ ] Caminho de `EXTRA_COMPONENT_DIRS` corrigido.
- [ ] Modelo INT8 copiado para `data/` e hash conferido.
- [ ] Nome e caminho do modelo atualizados.
- [ ] Entrada alterada para `int8 NCHW`.
- [ ] Saída alterada para 20 logits INT8.
- [ ] Resolver contém exatamente os sete operadores necessários.
- [ ] Pré-processamento reproduz o notebook.
- [ ] Ordem das 20 classes preservada.
- [ ] Scripts Python não possuem dependência de CIFAR-10.
- [ ] Hash da entrada coincide entre PC e ESP32.
- [ ] Classe e saída comparadas com a referência.
- [ ] Arena medida na placa.
- [ ] README e comandos atualizados.
- [ ] Credenciais removidas do código-fonte.
