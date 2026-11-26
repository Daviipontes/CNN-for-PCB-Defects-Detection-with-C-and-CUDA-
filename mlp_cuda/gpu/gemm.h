#ifndef GPU_GEMM_H
#define GPU_GEMM_H

#include "processamento.h"

// O que fazer com cada elemento de C depois da multiplicacao ("epilogo").
enum class Epilogo {
    Nenhum,             // C = soma
    ViesAtivacao,       // C = f(soma + vies[coluna])          -> propagacao
    GradienteAtivacao   // C = soma * f'(aux[linha, coluna])   -> retropropagacao (aux = saida da camada)
};

// C[M x N] = op(A) * op(B) na GPU (ponteiros ja na memoria da GPU), matrizes linha a linha:
//   transpA = false: A guardada M x K     transpA = true: A guardada K x M
//   transpB = false: B guardada K x N     transpB = true: B guardada N x K
// As versoes transpostas calculam A^T * dZ e dZ * W^T sem criar copias transpostas.
// 'vies' e usado so em Epilogo::ViesAtivacao e 'aux' so em Epilogo::GradienteAtivacao.
// Lanca std::runtime_error se o lancamento do kernel falhar.
void gemm(const float* A, const float* B, float* C, int M, int N, int K,
          bool transpA, bool transpB, Epilogo epilogo,
          const float* vies, const float* aux, Ativacao ativacao);

#endif
