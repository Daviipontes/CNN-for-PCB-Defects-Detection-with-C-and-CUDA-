#include "gpu/otimizador.h"

#include "gpu/cuda_util.h"

#include <cmath>

// Uma thread por parametro.
__global__ void kernelSGD(float* __restrict__ p, float* __restrict__ velocidade,
                          const float* __restrict__ g, int n, float taxa, float momento) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = momento * velocidade[i] + g[i];
        velocidade[i] = v;
        p[i] -= taxa * v;
    }
}

__global__ void kernelAdam(float* __restrict__ p, float* __restrict__ m, float* __restrict__ v,
                           const float* __restrict__ g, int n, float taxa, float beta1, float beta2,
                           float eps, float correcao1, float correcao2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float gi = g[i];
        const float mi = beta1 * m[i] + (1.0f - beta1) * gi;
        const float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] -= taxa * (mi / correcao1) / (sqrtf(vi / correcao2) + eps);
    }
}

void passoOtimizador(Otimizador tipo, float* p, float* m, float* v, const float* g,
                     int n, float taxa, int passo) {
    const int blocos = (n + 255) / 256;
    if (tipo == Otimizador::Adam) {
        const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
        // As correcoes de vies so dependem do passo: calculadas uma vez na CPU.
        const float c1 = 1.0f - std::pow(beta1, static_cast<float>(passo));
        const float c2 = 1.0f - std::pow(beta2, static_cast<float>(passo));
        kernelAdam<<<blocos, 256>>>(p, m, v, g, n, taxa, beta1, beta2, eps, c1, c2);
    } else {
        kernelSGD<<<blocos, 256>>>(p, m, g, n, taxa, 0.9f);
    }
    verificarCUDA(cudaGetLastError(), "Lancamento do kernel do otimizador");
}
