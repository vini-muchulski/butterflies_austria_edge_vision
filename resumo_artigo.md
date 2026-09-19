# Classificação de borboletas austríacas para visão computacional em dispositivos de borda

## Estado atual do trabalho

Este documento consolida a etapa inicial do estudo de classificação automática de espécies de borboletas com arquiteturas compactas de visão computacional. Até o momento, foram organizados o conjunto de dados, a pipeline de treinamento em PyTorch, o ajuste fino dos modelos MobileNetV2 e MobileViT-XXS, a avaliação no conjunto de teste e a conversão dos modelos para o formato TFLite com diferentes representações numéricas.

O objetivo experimental é avaliar o compromisso entre qualidade preditiva, número de parâmetros, custo computacional e tamanho do modelo exportado, considerando uma futura execução em dispositivos com recursos limitados. O MobileNetV2 foi adotado como principal baseline convolucional, enquanto o MobileViT-XXS foi incluído como uma arquitetura compacta que combina convoluções e mecanismos do tipo Transformer.

## Conjunto de dados

O conjunto utilizado contém 6.479 imagens RGB distribuídas em 20 classes de borboletas. Cada classe é representada por uma pasta, e o nome da pasta determina automaticamente o rótulo utilizado pelo `ImageFolder` do Torchvision. A distribuição original é desbalanceada: a menor classe contém 160 imagens e a maior contém 474 imagens, uma razão aproximada de 2,96 entre os extremos.

As classes presentes são:

1. `Aglais_Urticae`
2. `Anthocharis_Cardamines`
3. `Apatura_Iris`
4. `Argynnis_Paphia`
5. `Colias_Myrmidone`
6. `Gonepteryx_Rhamni`
7. `Inachis_Io`
8. `Iphiclides_Podalirius`
9. `Lycaena_Virgaureae`
10. `Lycaenidae`
11. `Maniola_Jurtina`
12. `Melanargia_Galathea`
13. `Nymphalis_Antiopa`
14. `Papilio_Machaon`
15. `Parnassius_Apollo`
16. `Pieris_Rapae`
17. `Polygonia_C-album`
18. `Vanessa_Atalanta`
19. `Vanessa_Cardui`
20. `Zerynthia_Polyxena`

### Organização e particionamento

Para impedir que os notebooks dependessem de uma divisão realizada em memória, foi criado um script específico para materializar a separação do conjunto em disco. As imagens originais foram mantidas intactas e copiadas para três diretórios independentes: `train`, `val` e `test`.

O particionamento foi estratificado por classe, com semente aleatória 42 e proporções de 80% para treinamento, 10% para validação e 10% para teste. A estrutura resultante segue o formato `split/classe/imagem`, diretamente compatível com `torchvision.datasets.ImageFolder`.

| Partição | Imagens | Proporção aproximada |
|---|---:|---:|
| Treinamento | 5.183 | 80% |
| Validação | 648 | 10% |
| Teste | 648 | 10% |
| Total | 6.479 | 100% |

A estratificação preservou aproximadamente a distribuição relativa de cada classe nas três partições. A distribuição detalhada foi:

| Classe | Total | Treino | Validação | Teste |
|---|---:|---:|---:|---:|
| `Aglais_Urticae` | 269 | 215 | 27 | 27 |
| `Anthocharis_Cardamines` | 452 | 361 | 46 | 45 |
| `Apatura_Iris` | 349 | 279 | 35 | 35 |
| `Argynnis_Paphia` | 395 | 316 | 39 | 40 |
| `Colias_Myrmidone` | 284 | 227 | 29 | 28 |
| `Gonepteryx_Rhamni` | 232 | 186 | 23 | 23 |
| `Inachis_Io` | 273 | 218 | 27 | 28 |
| `Iphiclides_Podalirius` | 297 | 238 | 30 | 29 |
| `Lycaena_Virgaureae` | 171 | 137 | 17 | 17 |
| `Lycaenidae` | 415 | 332 | 42 | 41 |
| `Maniola_Jurtina` | 407 | 326 | 40 | 41 |
| `Melanargia_Galathea` | 410 | 328 | 41 | 41 |
| `Nymphalis_Antiopa` | 160 | 128 | 16 | 16 |
| `Papilio_Machaon` | 242 | 194 | 24 | 24 |
| `Parnassius_Apollo` | 309 | 247 | 31 | 31 |
| `Pieris_Rapae` | 368 | 294 | 37 | 37 |
| `Polygonia_C-album` | 396 | 317 | 39 | 40 |
| `Vanessa_Atalanta` | 474 | 379 | 48 | 47 |
| `Vanessa_Cardui` | 332 | 266 | 33 | 33 |
| `Zerynthia_Polyxena` | 244 | 195 | 24 | 25 |

Os notebooks verificam se o mapeamento entre nomes de classes e índices é idêntico nas três partições. Essa validação reduz o risco de comparar predições e rótulos com codificações diferentes.

## Pipeline experimental

### Reprodutibilidade e carregamento

Foi utilizada a semente 42 nos módulos `random`, NumPy e PyTorch, incluindo as rotinas CUDA. Os carregadores utilizam lotes de 64 imagens, quatro processos auxiliares, memória fixada quando uma GPU está disponível e trabalhadores persistentes. O embaralhamento é habilitado somente no treinamento; validação e teste mantêm ordem determinística.

A configuração atual prioriza desempenho no backend cuDNN, com `benchmark=True` e `deterministic=False`. Portanto, a semente controla as principais fontes de aleatoriedade, mas não garante reprodutibilidade bit a bit entre execuções em GPU.

### Aumento de dados

O aumento de dados é aplicado exclusivamente às imagens de treinamento. Em ambos os modelos foram utilizadas inversões horizontal e vertical com probabilidade de 0,1 e transformação afim aleatória com rotação de até 10 graus, translação de até 10% em cada eixo e escala entre 0,9 e 1,1. Validação, teste e calibração utilizam somente transformações determinísticas.

### Compensação do desbalanceamento

O desbalanceamento foi tratado na função de perda. Para cada classe, calculou-se inicialmente o inverso de sua frequência relativa no conjunto de treinamento. Em seguida, aplicou-se a raiz quadrada desses valores para reduzir a agressividade da ponderação, e os pesos foram normalizados para média unitária.

A função objetivo foi a entropia cruzada multiclasse ponderada, com `label smoothing` igual a 0,05. Essa combinação busca aumentar a contribuição das classes menos representadas sem produzir pesos extremos e reduzir a confiança excessiva do classificador.

## Treinamento do MobileNetV2

### Arquitetura e pré-processamento

Foi utilizada a implementação `mobilenetv2_100` da biblioteca `timm`, inicializada com pesos pré-treinados. O classificador final foi substituído para produzir 20 logits, com `dropout` de 0,3. O modelo resultante possui 2.249.492 parâmetros.

As imagens são convertidas explicitamente para RGB, redimensionadas para 224 × 224 pixels e convertidas em tensores. Em seguida, são normalizadas com média `(0,485, 0,456, 0,406)` e desvio-padrão `(0,229, 0,224, 0,225)`, valores convencionais dos modelos pré-treinados no ImageNet.

### Otimização

Todo o modelo foi ajustado, mas com taxas de aprendizado diferentes para o backbone e para o novo classificador. O backbone recebeu taxa máxima de `1 × 10⁻⁴`, enquanto a cabeça classificadora recebeu `1 × 10⁻³`. Foi utilizado AdamW com decaimento de pesos de `1 × 10⁻⁴` e limitação da norma do gradiente em 1,0.

A taxa de aprendizado foi controlada pelo escalonador One-Cycle, configurado para até 40 épocas, `div_factor=100` e `final_div_factor=100`. O melhor estado foi definido pela maior acurácia de validação. O treinamento seria interrompido após oito épocas consecutivas sem melhora estrita nessa métrica.

### Convergência e desempenho

O MobileNetV2 atingiu sua melhor acurácia de validação na época 26, com 97,69%. Como não houve valor estritamente superior nas oito épocas seguintes, o treinamento foi encerrado antecipadamente na época 34. O melhor estado, e não o estado da última época, foi restaurado para a avaliação final.

No conjunto de teste, o modelo PyTorch atingiu:

| Métrica | Resultado |
|---|---:|
| Acurácia | 97,07% |
| Precisão macro | 97,13% |
| Recall macro | 96,98% |
| F1 macro | 97,00% |
| Precisão ponderada | 97,19% |
| Recall ponderado | 97,07% |
| F1 ponderado | 97,08% |

Três classes apresentaram F1 igual a 1,0: `Argynnis_Paphia`, `Iphiclides_Podalirius` e `Papilio_Machaon`. O menor F1 foi observado em `Gonepteryx_Rhamni`, com 91,67%, seguido por `Vanessa_Atalanta`, com 93,62%, e `Lycaena_Virgaureae`, com 94,12%. Mesmo nas classes mais difíceis, o desempenho permaneceu elevado.

## Treinamento do MobileViT-XXS

### Arquitetura e pré-processamento

Foi utilizado o modelo pré-treinado `apple/mobilevit-xx-small`, carregado pela biblioteca Transformers. A camada classificadora original, destinada a 1.000 classes, foi reinicializada para as 20 classes do conjunto estudado. O modelo foi encapsulado por um módulo PyTorch que retorna diretamente os logits. A configuração resultante possui 957.444 parâmetros.

As imagens de treinamento são redimensionadas para 288 pixels e submetidas a um recorte aleatório de 256 × 256 pixels. Validação e teste utilizam recorte central de 256 × 256 pixels. Após a conversão para tensor no intervalo `[0, 1]`, os canais RGB são invertidos para BGR, conforme o pré-processamento esperado por essa implementação do MobileViT.

### Otimização e resultados

O MobileViT-XXS utilizou o mesmo protocolo de otimização do MobileNetV2: AdamW, taxas máximas de `1 × 10⁻⁴` no backbone e `1 × 10⁻³` no classificador, decaimento de pesos de `1 × 10⁻⁴`, limitação de gradiente em 1,0, One-Cycle, máximo de 40 épocas e paciência de oito épocas.

A melhor acurácia de validação foi 97,53%, obtida na época 26. O treinamento também foi interrompido na época 34. No teste, o modelo PyTorch atingiu:

| Métrica | Resultado |
|---|---:|
| Acurácia | 97,22% |
| Precisão macro | 97,27% |
| Recall macro | 97,31% |
| F1 macro | 97,21% |
| Precisão ponderada | 97,35% |
| Recall ponderado | 97,22% |
| F1 ponderado | 97,22% |

O MobileViT-XXS apresentou acurácia de teste 0,15 ponto percentual superior à do MobileNetV2, porém essa diferença corresponde a apenas uma predição no conjunto de 648 imagens e não deve ser interpretada como superioridade estatística. Seu número de parâmetros é aproximadamente 57,4% menor. Por outro lado, a conversão registrou cerca de 535 milhões de operações multiply-accumulate para o MobileViT-XXS, contra aproximadamente 303 milhões para o MobileNetV2, indicando que menor quantidade de parâmetros não implica necessariamente menor custo computacional.

## Conversão para TFLite e quantização pós-treinamento

Após o ajuste fino, os dois modelos foram convertidos de PyTorch para TFLite por meio do LiteRT-Torch. Para calibração foram selecionadas, com semente 42, até 100 imagens da partição de treinamento, processadas sem aumento aleatório. Dessa forma, nenhuma imagem de validação ou teste foi utilizada na calibração.

Foram geradas três versões de cada arquitetura:

- modelo TFLite em ponto flutuante de 32 bits;
- pesos de 8 bits e ativações de 16 bits;
- pesos e ativações de 8 bits.

### Resultados do MobileNetV2 exportado

| Modelo | Entrada e saída | Tamanho | Redução | Acurácia |
|---|---|---:|---:|---:|
| TFLite FP32 | `float32` | 8,63 MB | — | 97,07% |
| Pesos 8 bits/ativações 16 bits | `int16` | 2,75 MB | 3,1× menor | 97,22% |
| Pesos 8 bits/ativações 8 bits | `int8` | 2,68 MB | 3,2× menor | 97,22% |

A conversão em ponto flutuante preservou exatamente a acurácia observada no modelo PyTorch. As duas versões quantizadas apresentaram 97,22%, diferença positiva de 0,15 ponto percentual em relação ao modelo de ponto flutuante. Essa pequena variação equivale a uma imagem no conjunto de teste e deve ser interpretada como uma flutuação numérica discreta, não como evidência de ganho de generalização causado pela quantização.

O resultado principal do MobileNetV2 é a manutenção integral da qualidade preditiva após a redução do modelo de 8,63 MB para aproximadamente 2,7 MB. Entre as configurações avaliadas, a representação integral de 8 bits apresentou o menor arquivo, 2,68 MB, sem degradação mensurável de acurácia.

### Resultados do MobileViT-XXS exportado

| Modelo | Entrada e saída | Tamanho | Redução | Acurácia |
|---|---|---:|---:|---:|
| TFLite FP32 | `float32` | 4,33 MB | — | 97,38% |
| Pesos 8 bits/ativações 16 bits | `int16` | 1,89 MB | 2,3× menor | 96,76% |
| Pesos 8 bits/ativações 8 bits | `int8` | 1,83 MB | 2,4× menor | 9,57% |

A versão de ponto flutuante preservou o desempenho geral, com diferença de apenas 0,16 ponto percentual em relação ao modelo PyTorch. A configuração com ativações de 16 bits sofreu redução de 0,62 ponto percentual em relação ao TFLite FP32, mantendo desempenho elevado.

Em contraste, a configuração integral de 8 bits apresentou colapso de acurácia para 9,57%. O relatório por classe mostra concentração das predições em poucas categorias e F1 ponderado de apenas 5,11%. Esse comportamento evidencia sensibilidade da arquitetura MobileViT-XXS à configuração de quantização integral utilizada e exige investigação específica antes de sua adoção nessa representação.

### Mitigação do colapso INT8 por agrupamento estrutural

Para investigar o colapso, foram medidas as ativações por canal na saída da convolução de expansão `1 × 1` dos sete módulos `MobileViTInvertedResidual`. Cada um desses módulos, denominado bloco neste estudo, contém uma convolução de expansão `1 × 1`, uma convolução depthwise `3 × 3`, uma convolução de redução `1 × 1` e, quando as dimensões permitem, uma conexão residual. Para cada canal, a escala foi estimada pela diferença entre seus valores máximo e mínimo observados durante a calibração, dividida pelos 255 intervalos representáveis em 8 bits.

A análise revelou forte heterogeneidade entre canais de um mesmo tensor. As razões entre a maior e a menor escala chegaram a 327,61 no bloco 1, 874,36 no bloco 2, 318,00 no bloco 5 e 487,58 no bloco 7. Na quantização por tensor, todos os canais compartilham a mesma escala. Dessa forma, canais de grande amplitude determinam o intervalo quantizado, enquanto canais de menor amplitude passam a ocupar poucos códigos inteiros e perdem resolução numérica.

Para reduzir esse efeito, foi aplicada a configuração estrutural `g8_4_4`. Essa notação indica que os canais internos do bloco 1 foram divididos em oito grupos, enquanto os blocos 2 e 5 foram divididos em quatro grupos cada. O bloco 1 possui 32 canais de expansão, resultando em oito grupos de quatro canais. O bloco 2 também possui 32 canais, divididos em quatro grupos de oito canais, e o bloco 5 possui 48 canais, divididos em quatro grupos de 12 canais.

Os canais de cada bloco foram ordenados por suas escalas observadas e particionados em grupos de tamanhos aproximadamente iguais. Consequentemente, cada grupo reuniu canais com faixas numéricas semelhantes, independentemente de suas posições originais no tensor. O termo grupo, nesse contexto, representa um subconjunto de canais internos e não uma divisão das imagens ou dos lotes de entrada.

Cada bloco selecionado foi então substituído por ramificações paralelas, uma para cada grupo. Em cada ramificação, foram preservados somente os filtros correspondentes na convolução de expansão, os mesmos canais na convolução depthwise e as colunas correspondentes dos pesos da convolução de redução. Cada ramificação continuou produzindo todos os canais de saída do bloco, e suas saídas foram somadas antes da aplicação única da conexão residual.

Como a convolução de redução era seguida por normalização em lote, seus parâmetros foram inicialmente incorporados aos pesos e ao bias da convolução. O bias resultante foi dividido pelo número de grupos e atribuído a cada ramificação. Assim, a soma das ramificações reconstrói um único bias completo, evitando que esse termo seja multiplicado pelo número de grupos.

Essa transformação procura preservar a função original do bloco, pois a convolução de redução é linear em relação aos canais de entrada. Ao mesmo tempo, as ramificações passam a gerar tensores independentes durante a quantização. Cada grupo pode, portanto, receber uma escala INT8 mais apropriada à amplitude de seus canais, reduzindo a influência de valores extremos sobre canais de menor magnitude.

O experimento reutilizou o checkpoint treinado do MobileViT-XXS, sem novo treinamento ou QAT. A calibração foi estratificada, com 50 imagens por classe e 1.000 imagens no total. Foram exportados os modelos original e agrupado em FP32 e posteriormente quantizados com pesos e ativações de 8 bits. Ambos os modelos quantizados apresentaram entrada e saída `int8`, sem operadores incompatíveis ou operações explícitas de desquantização.

| Modelo | Acurácia de validação | Acurácia de teste | F1 macro no teste | Tamanho |
|---|---:|---:|---:|---:|
| MobileViT-XXS original FP32 | 97,38% | 97,38% | 97,39% | 4,33 MB |
| MobileViT-XXS original INT8, calibração estratificada | 10,96% | 10,80% | 6,05% | 1,83 MB |
| MobileViT-XXS `g8_4_4` FP32 | 97,38% | 97,38% | 97,39% | 4,47 MB |
| MobileViT-XXS `g8_4_4` INT8 | 89,20% | 90,43% | 90,23% | 1,98 MB |

O agrupamento elevou a acurácia INT8 de 10,80% para 90,43% no teste, um ganho de 79,63 pontos percentuais, e recuperou aproximadamente 92% da acurácia perdida pela quantização do modelo original. A calibração estratificada isoladamente não impediu o colapso, como demonstra o resultado do modelo original quantizado sob a mesma calibração. Portanto, o ganho é atribuído principalmente à separação estrutural dos canais em faixas de magnitude mais homogêneas.

A transformação, entretanto, não atingiu equivalência FP32 estrita no PyTorch. A concordância top-1 entre o modelo original e o agrupado foi 99,85%, uma das 648 imagens de validação mudou de classificação e o erro absoluto máximo entre logits foi 0,0371. Os modelos TFLite FP32 apresentaram a mesma acurácia, mas a diferença numérica impede caracterizar a transformação atual como matematicamente exata. A alteração da ordem das somas em ponto flutuante é uma possível fonte dessa diferença e deverá ser investigada em experimentos posteriores.

## Comparação consolidada

| Arquitetura e representação | Parâmetros | Tamanho | Acurácia de teste |
|---|---:|---:|---:|
| MobileNetV2 PyTorch | 2.249.492 | — | 97,07% |
| MobileNetV2 TFLite FP32 | 2.249.492 | 8,63 MB | 97,07% |
| MobileNetV2 TFLite 8/16 bits | 2.249.492 | 2,75 MB | 97,22% |
| MobileNetV2 TFLite 8/8 bits | 2.249.492 | 2,68 MB | 97,22% |
| MobileViT-XXS PyTorch | 957.444 | — | 97,22% |
| MobileViT-XXS TFLite FP32 | 957.444 | 4,33 MB | 97,38% |
| MobileViT-XXS TFLite 8/16 bits | 957.444 | 1,89 MB | 96,76% |
| MobileViT-XXS TFLite 8/8 bits | 957.444 | 1,83 MB | 9,57% |
| MobileViT-XXS TFLite 8/8 bits `g8_4_4` | 957.444 | 1,98 MB | 90,43% |

Os modelos em ponto flutuante apresentaram desempenho praticamente equivalente. O MobileViT-XXS oferece menor número de parâmetros e menor arquivo TFLite, enquanto o MobileNetV2 apresenta menor estimativa de operações e comportamento substancialmente mais estável sob quantização integral. O agrupamento `g8_4_4` restaurou grande parte do desempenho INT8 do MobileViT-XXS, mas sua acurácia permaneceu 6,94 pontos percentuais abaixo da respectiva versão FP32. Considerando simultaneamente acurácia, tamanho e robustez da conversão, o MobileNetV2 integral de 8 bits permanece, até esta etapa, o baseline mais consistente para implantação em borda.

## Limitações da etapa atual

Os resultados correspondem a uma única divisão estratificada e uma única execução de treinamento por arquitetura. Não foram calculados intervalos de confiança, variância entre sementes ou validação cruzada. Portanto, diferenças inferiores a um ponto percentual devem ser tratadas com cautela.

O conjunto apresenta desbalanceamento moderado, tratado por ponderação da função de perda, mas ainda não foi realizada uma análise sistemática do efeito isolado dessa estratégia. Também não foram medidos tempo de inferência, consumo de memória ou energia em hardware de borda real. O tamanho do arquivo e a contagem estimada de operações são indicadores de eficiência, mas não substituem medições de latência no dispositivo-alvo.

A calibração dos experimentos iniciais foi realizada com 100 imagens selecionadas aleatoriamente do treinamento. No experimento de agrupamento estrutural foram utilizadas 1.000 imagens, com 50 exemplos por classe. Embora a calibração estratificada isoladamente não tenha evitado o colapso, estudos posteriores devem separar sistematicamente os efeitos do tamanho, da representatividade e da estratégia de agrupamento.

O agrupamento `g8_4_4` foi avaliado em uma única configuração estrutural. Além disso, a transformação não satisfez o critério de equivalência FP32 estrita. Ainda devem ser investigadas outras combinações de blocos e números de grupos, a estabilidade entre diferentes subconjuntos de calibração e a origem da divergência numérica introduzida pela soma das ramificações.

## Conclusões parciais

A pipeline construída permite comparação controlada entre arquiteturas compactas usando as mesmas divisões de dados, critérios de treinamento e métricas. MobileNetV2 e MobileViT-XXS alcançaram aproximadamente 97% de acurácia no conjunto de teste, confirmando que ambas as arquiteturas conseguem representar adequadamente as 20 classes avaliadas.

O MobileNetV2 apresentou o resultado mais equilibrado para o objetivo de visão computacional em borda: 97,22% de acurácia após quantização integral de 8 bits, arquivo de 2,68 MB e redução de 3,2 vezes em relação ao TFLite FP32. A quantização INT8 convencional do MobileViT-XXS sofreu colapso, mas o agrupamento estrutural `g8_4_4` elevou sua acurácia para 90,43%, demonstrando que a heterogeneidade das ativações por canal era uma causa relevante da degradação. Esse resultado amplia a viabilidade do MobileViT-XXS em INT8, embora permaneçam uma diferença de 6,94 pontos percentuais em relação ao FP32 e a necessidade de obter equivalência numérica mais rigorosa. Até o momento, os experimentos indicam que a escolha da arquitetura deve considerar não apenas a acurácia em ponto flutuante e o número de parâmetros, mas também a distribuição interna das ativações e a estabilidade do modelo durante a conversão e a quantização destinadas ao ambiente de implantação.
