#ifndef GPU_ATIVACOES_CUH
#define GPU_ATIVACOES_CUH

// Funcoes de dispositivo: so podem ser incluidas em arquivos .cu (compilados pelo nvcc).

#include "processamento.h"

#include <cuda_runtime.h>

__device__ __forceinline__ float ativar(float x, Ativacao a) {
    switch (a) {
        case Ativacao::Sigmoide: return 1.0f / (1.0f + expf(-x));
        case Ativacao::Tanh:     return tanhf(x);
        case Ativacao::ReLU:     return x > 0.0f ? x : 0.0f;
        default:                 return x;  // Identidade
    }
}

// Derivada escrita em funcao da SAIDA y = f(z). Por isso basta guardar as ativacoes
// (e nao os valores antes da ativacao) para o backpropagation.
__device__ __forceinline__ float derivadaPelaSaida(float y, Ativacao a) {
    switch (a) {
        case Ativacao::Sigmoide: return y * (1.0f - y);
        case Ativacao::Tanh:     return 1.0f - y * y;
        case Ativacao::ReLU:     return y > 0.0f ? 1.0f : 0.0f;
        default:                 return 1.0f;  // Identidade
    }
}

#endif
