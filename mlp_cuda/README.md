# Rede neural MLP: C++ + CUDA

O programa classifica defeitos de placas de circuito impresso (PCB_DATASET, `archive.zip`) em 6 classes (`Missing_hole`, `Mouse_bite`, `Open_circuit`, `Short`, `Spur`, `Spurious_copper`). Um perceptron multicamadas (MLP) e treinado inteiramente na GPU e as previsoes do conjunto de teste sao gravadas em `saida.csv`.

## Dataset

O programa le o `archive.zip` direto, sem extrair nada em disco (o zip tem ~2 GB e e lido por entrada). Para cada defeito anotado nos XML de `Annotations/`, recorta uma janela com o dobro do tamanho da caixa, converte para tons de cinza, reduz para 32x32 e padroniza (media 0, desvio 1). A divisao 80/20 e feita por imagem e estratificada por classe. As imagens rotacionadas (`rotation/`) nao sao usadas. A leitura (zip + JPEG) e feita na CPU, uma vez, antes do treino.

## Requisitos

- CUDA Toolkit com `nvcc`
- g++, make e zlib (`zlib1g-dev`)
- Python 3 (apenas para os testes sem GPU)
- `third_party/stb_image.h` (decodificador JPEG/PNG de dominio publico, ja incluido)

## Compilar

```bash
make
```

O `Makefile` compila cada `.cpp` com `g++` e cada `.cu` com `nvcc` (sempre com `-I.`, pois os includes partem da raiz do projeto), guarda os objetos em `obj/` e liga tudo com `nvcc ... -lz -o mlp_cuda`. Arquivos novos em `dataset/` ou `gpu/` entram no build sem editar o `Makefile`.

Para gerar codigo para a sua GPU: `make ARCH=sm_86` (troque pelo valor da sua placa).

## Executar

```bash
./mlp_cuda ../archive.zip 100 32
```

Os tres argumentos sao opcionais: caminho do zip (padrao: `../archive.zip`), numero de epocas (padrao: 100) e amostras por lote (padrao: 32). O programa mostra a perda, a acuracia no treino e no teste, a matriz de confusao e o tempo de treino.

A rede e `1024 -> 128 -> 64 -> 6` (tanh nas ocultas, saida linear). Os rotulos viram vetores one-hot e a perda continua sendo MSE; a classe prevista e a maior saida. 
A saida tem nome fixo, `saida.csv`, com as colunas `indice,classe_real,classe_prevista` (conjunto de teste).

## Arquivos

CPU (compilada com g++):

- `main.cpp`: so orquestra: le os argumentos, carrega o dataset, treina, avalia e mede o tempo.
- `avaliacao.h` / `avaliacao.cpp`: classe prevista, acuracia, matriz de confusao e gravacao do CSV.
- `dataset/dados.h`: struct `Dados` (entradas, alvos one-hot, rotulos).
- `dataset/zip_reader.*`: `LeitorZip`, leitura de entradas do zip direto do disco (zlib, zip64).
- `dataset/xml_pcb.*`: `lerAnotacao`, extrai `<filename>` e as caixas `<bndbox>` do XML.
- `dataset/imagem.*`: `recortesDaImagem`, decodifica a imagem e gera um recorte padronizado por caixa.
- `dataset/dataset_pcb.*`: `carregarPCB`, junta os modulos acima e faz a divisao treino/teste por imagem.
- `third_party/stb_image.h`: biblioteca de terceiros (dominio publico) para decodificar JPEG/PNG.

GPU (compilada com nvcc):

- `processamento.h`: interface entre a CPU e a GPU, sem nenhum tipo CUDA.
- `processamento.cu`: implementa a interface (`criarMLP`, `treinarMLP`, `preverMLP`): o laco de treino.
- `gpu/cuda_util.h`: `verificarCUDA` e `MemoriaGPU<T>` (buffer que se libera sozinho).
- `gpu/ativacoes.cuh`: funcoes de ativacao e derivadas (dispositivo).
- `gpu/gemm.*`: produto de matrizes com tiles e epilogo fundido.
- `gpu/kernels_aux.*`: soma por coluna, perda MSE e montagem do lote.
- `gpu/otimizador.*`: SGD e Adam.
- `gpu/rede.*`: layout dos parametros, buffers de ativacao, propagacao e validacoes.

## Observacoes

- Pesos, ativacoes, gradientes e o conjunto de dados sao copiados para a GPU uma unica vez. Durante o treino, a CPU so espera a GPU uma vez por epoca, para ler a perda.
- Os lotes sao embaralhados a cada epoca: somente os indices sao enviados, e o lote e montado na propria GPU. O ultimo lote pode ter menos amostras que os anteriores.
- O produto de matrizes usa tiles em memoria compartilhada (64x64 por bloco) e cada thread acumula 4x4 resultados em registradores. As versoes transpostas calculam `A^T * dZ` e `dZ * W^T` na retropropagacao sem criar copias transpostas.
- Passos elemento a elemento sao fundidos: `f(X * W + b)` e um unico lancamento na propagacao, e `(dZ * W^T) * f'(A)` e um unico lancamento na retropropagacao. A perda MSE ja devolve o gradiente da saida.
- Todos os parametros ficam em um buffer continuo, entao um unico lancamento do otimizador (Adam ou SGD com momento) atualiza a rede inteira.
- O tempo medido inclui apenas o treino. O contexto CUDA e criado antes, em `inicializarCUDA()`.

## Testes sem GPU

```bash
make testes
```

A pasta `testes/` tem uma emulacao do runtime CUDA na CPU que executa os kernels reais de `processamento.cu` e `gpu/*.cu`. Cada thread CUDA e uma corrotina interrompida em todo `__syncthreads()`, entao a falta de uma barreira ou um `return` antes dela faz os testes falharem.

Os testes:

- comparam cada kernel com uma versao simples em double;
- conferem que o treino acompanha uma implementacao de referencia passo a passo;
- validam as entradas;
- verificam a convergencia em `sin^2`;
- executam o `main.cpp` emulado com um zip sintetico pequeno (mesma estrutura do `archive.zip`, imagens PNG), o que exercita o leitor de zip, o XML e os recortes.

Eles conferem a correcao do codigo, nao o desempenho na GPU.
