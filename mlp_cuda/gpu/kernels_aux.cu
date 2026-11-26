#include "gpu/kernels_aux.h"

#include "gpu/cuda_util.h"

// Cada bloco cuida de 32 colunas com 8 grupos de linhas; threads consecutivas leem colunas
// consecutivas (leitura coalescida) e as 8 somas parciais sao combinadas na memoria compartilhada.
__global__ void kernelSomaColunas(const float* __restrict__ X, float* __restrict__ saida,
                                  int linhas, int colunas) {
    __shared__ float parcial[8][33];
    const int tx = threadIdx.x % 32;
    const int ty = threadIdx.x / 32;
    const int c = blockIdx.x * 32 + tx;

    float s = 0.0f;
    if (c < colunas)
        for (int r = ty; r < linhas; r += 8) s += X[(size_t)r * colunas + c];
    parcial[ty][tx] = s;
    __syncthreads();

    if (ty == 0 && c < colunas) {
        float total = 0.0f;
        #pragma unroll
        for (int g = 0; g < 8; ++g) total += parcial[g][tx];
        saida[c] = total;
    }
}

void somaColunas(const float* X, float* saida, int linhas, int colunas) {
    kernelSomaColunas<<<(colunas + 31) / 32, 256>>>(X, saida, linhas, colunas);
    verificarCUDA(cudaGetLastError(), "Lancamento do kernel de soma por coluna");
}

// Cada bloco soma seus erros quadraticos na memoria compartilhada (reducao em arvore) e uma
// unica thread por bloco faz o atomicAdd no total.
constexpr int THREADS_PERDA = 256;

__global__ void kernelPerdaMSE(const float* __restrict__ previsto, const float* __restrict__ alvo,
                               float* __restrict__ delta, int n, float escala, float* somaErros) {
    __shared__ float erros[THREADS_PERDA];
    const int t = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + t;

    float erro2 = 0.0f;
    if (i < n) {
        const float diferenca = previsto[i] - alvo[i];
        delta[i] = escala * diferenca;
        erro2 = diferenca * diferenca;
    }
    erros[t] = erro2;
    __syncthreads();

    for (int passo = THREADS_PERDA / 2; passo > 0; passo >>= 1) {
        if (t < passo) erros[t] += erros[t + passo];
        __syncthreads();
    }
    if (t == 0) atomicAdd(somaErros, erros[0]);
}

void perdaMSE(const float* previsto, const float* alvo, float* delta, int n, float* somaErros) {
    const float escala = 2.0f / static_cast<float>(n);
    kernelPerdaMSE<<<(n + THREADS_PERDA - 1) / THREADS_PERDA, THREADS_PERDA>>>(previsto, alvo, delta, n, escala, somaErros);
    verificarCUDA(cudaGetLastError(), "Lancamento do kernel de perda");
}

__global__ void kernelReunirLinhas(const float* __restrict__ origem, const int* __restrict__ indices,
                                   float* __restrict__ destino, int linhas, int colunas) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < linhas * colunas) {
        const int r = e / colunas;
        const int c = e % colunas;
        destino[e] = origem[(size_t)indices[r] * colunas + c];
    }
}

void reunirLinhas(const float* origem, const int* indices, float* destino, int linhas, int colunas) {
    const int n = linhas * colunas;
    kernelReunirLinhas<<<(n + 255) / 256, 256>>>(origem, indices, destino, linhas, colunas);
    verificarCUDA(cudaGetLastError(), "Lancamento do kernel de montagem do lote");
}
