#ifndef GPU_KERNELS_AUX_H
#define GPU_KERNELS_AUX_H

// Kernels auxiliares do treino. Todos os ponteiros apontam para a memoria da GPU e todas as
// funcoes lancam std::runtime_error se o lancamento do kernel falhar.

// saida[c] = soma das linhas de X na coluna c (gradiente do vies). X: [linhas x colunas].
void somaColunas(const float* X, float* saida, int linhas, int colunas);

// Perda MSE fundida com o gradiente da saida (camada de saida linear):
//   delta = 2 * (previsto - alvo) / n
// e acumula a soma dos erros quadraticos em *somaErros (tem que estar zerado antes da 1a chamada).
void perdaMSE(const float* previsto, const float* alvo, float* delta, int n, float* somaErros);

// destino[r, :] = origem[indices[r], :]  (monta o lote embaralhado direto na GPU).
void reunirLinhas(const float* origem, const int* indices, float* destino, int linhas, int colunas);

#endif
