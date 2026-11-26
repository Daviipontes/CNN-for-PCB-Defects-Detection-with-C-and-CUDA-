#include "gpu/gemm.h"

#include "gpu/ativacoes.cuh"
#include "gpu/cuda_util.h"

// Regra seguida pelos kernels que usam __syncthreads(): nenhuma thread sai antes da hora.
// Threads fora da matriz continuam o laco e apenas pulam leituras/escritas, para que todas
// cheguem as mesmas barreiras.

// Cada bloco calcula um pedaco 64x64 de C, percorrendo K em fatias de 8.
// 256 threads = 16 x 16; cada thread acumula 4x4 valores em registradores.
// A thread (tx, ty) cuida das linhas ty + 16*i e colunas tx + 16*j (i, j = 0..3): com esse
// espacamento as leituras da memoria compartilhada nao tem conflito de banco e threads vizinhas
// gravam colunas vizinhas de C (escrita coalescida).
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 8;
constexpr int THREADS_GEMM = 256;
constexpr int PREENCHIMENTO = 4;  // linhas de 68 floats: as escritas nos tiles caem em 32 bancos diferentes

template <bool TRANSP_A, bool TRANSP_B>
__global__ void __launch_bounds__(THREADS_GEMM)
kernelGemm(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
           int M, int N, int K, Epilogo epilogo,
           const float* __restrict__ vies, const float* __restrict__ aux, Ativacao ativacao) {

    // Tiles guardados "por k": As[k][m] e Bs[k][n].
    __shared__ float As[TILE_K][TILE_M + PREENCHIMENTO];
    __shared__ float Bs[TILE_K][TILE_N + PREENCHIMENTO];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int linha0 = blockIdx.y * TILE_M;
    const int coluna0 = blockIdx.x * TILE_N;

    float soma[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) soma[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {

        // 1) Carga cooperativa do tile de A (64 x 8 = 512 valores, 2 por thread).
        //    Threads consecutivas leem enderecos consecutivos: ao longo de k quando A e M x K,
        //    ao longo de m quando A esta guardada transposta (K x M).
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int e = tid + r * THREADS_GEMM;
            const int m = TRANSP_A ? (e % TILE_M) : (e / TILE_K);
            const int k = TRANSP_A ? (e / TILE_M) : (e % TILE_K);
            const int gm = linha0 + m;
            const int gk = k0 + k;
            float v = 0.0f;  // zero fora da matriz: nao altera as somas
            if (gm < M && gk < K)
                v = TRANSP_A ? A[(size_t)gk * M + gm] : A[(size_t)gm * K + gk];
            As[k][m] = v;
        }

        // 2) Mesma coisa para o tile de B (8 x 64).
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int e = tid + r * THREADS_GEMM;
            const int n = TRANSP_B ? (e / TILE_K) : (e % TILE_N);
            const int k = TRANSP_B ? (e % TILE_K) : (e / TILE_N);
            const int gn = coluna0 + n;
            const int gk = k0 + k;
            float v = 0.0f;
            if (gn < N && gk < K)
                v = TRANSP_B ? B[(size_t)gn * K + gk] : B[(size_t)gk * N + gn];
            Bs[k][n] = v;
        }

        __syncthreads();  // o tile precisa estar completo antes de qualquer leitura

        // 3) Produtos externos: cada valor lido da memoria compartilhada e reutilizado 4 vezes.
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[k][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) soma[i][j] += a[i] * b[j];
        }

        __syncthreads();  // ninguem sobrescreve o tile enquanto outras threads ainda leem
    }

    // 4) Epilogo fundido e escrita com verificacao de limites.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int linha = linha0 + ty + 16 * i;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int coluna = coluna0 + tx + 16 * j;
            if (linha < M && coluna < N) {
                float v = soma[i][j];
                if (epilogo == Epilogo::ViesAtivacao)
                    v = ativar(v + vies[coluna], ativacao);
                else if (epilogo == Epilogo::GradienteAtivacao)
                    v *= derivadaPelaSaida(aux[(size_t)linha * N + coluna], ativacao);
                C[(size_t)linha * N + coluna] = v;
            }
        }
    }
}

void gemm(const float* A, const float* B, float* C, int M, int N, int K,
          bool transpA, bool transpB, Epilogo epilogo,
          const float* vies, const float* aux, Ativacao ativacao) {
    // O grid e calculado pelo tamanho da SAIDA (M x N).
    const dim3 bloco(THREADS_GEMM);
    const dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    if (!transpA && !transpB)
        kernelGemm<false, false><<<grid, bloco>>>(A, B, C, M, N, K, epilogo, vies, aux, ativacao);
    else if (transpA && !transpB)
        kernelGemm<true, false><<<grid, bloco>>>(A, B, C, M, N, K, epilogo, vies, aux, ativacao);
    else if (!transpA && transpB)
        kernelGemm<false, true><<<grid, bloco>>>(A, B, C, M, N, K, epilogo, vies, aux, ativacao);
    else
        kernelGemm<true, true><<<grid, bloco>>>(A, B, C, M, N, K, epilogo, vies, aux, ativacao);
    verificarCUDA(cudaGetLastError(), "Lancamento do kernel GEMM");
}
