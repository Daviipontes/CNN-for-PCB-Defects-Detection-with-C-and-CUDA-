#ifndef GPU_REDE_H
#define GPU_REDE_H

#include "gpu/cuda_util.h"
#include "processamento.h"

#include <vector>

// Estrutura da rede na GPU: onde ficam os parametros e as ativacoes de cada camada.

// Posicao de cada W e b dentro do vetor unico de parametros [W0 b0 W1 b1 ...].
struct Deslocamentos {
    std::vector<size_t> W, b;
    size_t total = 0;
};

Deslocamentos calcularDeslocamentos(const std::vector<int>& camadas);

// Buffers de ativacao de uma rede para lotes de ate 'lote' amostras.
// ativacoes[l]: saida da camada l, [lote x camadas[l]]; ativacoes[0] e a entrada do lote.
struct AtivacoesGPU {
    std::vector<MemoriaGPU<float>> ativacoes;
    AtivacoesGPU(const std::vector<int>& camadas, int lote);
};

// Propagacao para 'linhas' amostras ja copiadas em ativacoes[0]:
//   A[l+1] = f(A[l] * W[l] + b[l])   -> um unico lancamento de GEMM por camada
// 'parametros' esta na GPU. A ultima camada e sempre linear.
void propagar(const std::vector<int>& camadas, Ativacao ativacao, const Deslocamentos& d,
              const float* parametros, AtivacoesGPU& rede, int linhas);

// Validacoes de entrada (lancam std::runtime_error).
void validarModelo(const ModeloMLP& modelo);
int contarAmostras(const std::vector<float>& dados, int porAmostra, const char* nome);

#endif
