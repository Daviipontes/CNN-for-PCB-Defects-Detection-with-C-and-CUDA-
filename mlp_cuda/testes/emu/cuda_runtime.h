// Emulacao minima do runtime CUDA na CPU, usada SOMENTE pelos testes (nao precisa de GPU).
//
// Modelo de execucao: os blocos de um lancamento rodam um depois do outro. Dentro do bloco, cada
// thread CUDA e uma corrotina (ucontext) com pilha propria. O escalonador roda a thread 0 ate ela
// chegar em __syncthreads(), depois a thread 1, ... e entao todas seguem ate a proxima barreira.
// Essa ordem e proposital: se faltar um __syncthreads(), a thread 0 le dados que as outras ainda
// nao escreveram e o teste falha. Se algumas threads terminarem enquanto outras esperam numa
// barreira (return antes de __syncthreads), o emulador aborta.
//
// __shared__ vira `static`: uma copia por kernel, compartilhada pelas threads do bloco e
// reaproveitada pelo bloco seguinte (como a memoria compartilhada real, nao e zerada).
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
#include <ucontext.h>

// ---- CUDA keywords ----
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__ static

struct uint3 { unsigned int x, y, z; };
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};

namespace emu {
inline uint3& thread_idx() { static uint3 v; return v; }
inline uint3& block_idx() { static uint3 v; return v; }
inline dim3& block_dim() { static dim3 v; return v; }
inline dim3& grid_dim() { static dim3 v; return v; }
}
#define threadIdx (emu::thread_idx())
#define blockIdx (emu::block_idx())
#define blockDim (emu::block_dim())
#define gridDim (emu::grid_dim())

namespace emu {

struct State {
    ucontext_t scheduler;
    std::vector<ucontext_t> ctx;
    std::vector<std::vector<char>> stacks;
    std::vector<char> done;
    int current = 0;
    const std::function<void()>* body = nullptr;
    long long launches = 0;
};
inline State& state() { static State s; return s; }

inline void trampoline() {
    State& s = state();
    (*s.body)();
    s.done[s.current] = 1;
    swapcontext(&s.ctx[s.current], &s.scheduler);
}

inline void sync() {
    State& s = state();
    swapcontext(&s.ctx[s.current], &s.scheduler);
}

inline void launch(dim3 grid, dim3 block, const std::function<void()>& body) {
    State& s = state();
    const int nt = block.x * block.y * block.z;
    if (nt <= 0 || nt > 1024) { std::fprintf(stderr, "emu: invalid block size %d\n", nt); std::abort(); }
    if ((int)s.ctx.size() < nt) {
        s.ctx.resize(nt);
        s.stacks.resize(nt);
        for (auto& st : s.stacks) if (st.empty()) st.resize(64 * 1024);
    }
    s.done.assign(nt, 0);
    s.body = &body;
    block_dim() = block;
    grid_dim() = grid;
    ++s.launches;

    for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned bx = 0; bx < grid.x; ++bx) {
        block_idx() = {bx, by, bz};
        for (int t = 0; t < nt; ++t) {
            getcontext(&s.ctx[t]);
            s.ctx[t].uc_stack.ss_sp = s.stacks[t].data();
            s.ctx[t].uc_stack.ss_size = s.stacks[t].size();
            s.ctx[t].uc_link = nullptr;
            makecontext(&s.ctx[t], trampoline, 0);
            s.done[t] = 0;
        }
        while (true) {
            int finished = 0;
            for (int t = 0; t < nt; ++t) {
                if (s.done[t]) { ++finished; continue; }
                s.current = t;
                thread_idx() = {t % block.x, (t / block.x) % block.y, t / (block.x * block.y)};
                swapcontext(&s.scheduler, &s.ctx[t]);
                if (s.done[t]) ++finished;
            }
            if (finished == nt) break;
            // Some threads are parked at a barrier: all of them must be parked, none finished.
            int done_now = 0;
            for (int t = 0; t < nt; ++t) done_now += s.done[t];
            if (done_now != 0) {
                std::fprintf(stderr, "emu: barrier divergence (%d of %d threads exited while others "
                                     "wait at __syncthreads)\n", done_now, nt);
                std::abort();
            }
        }
    }
}

} // namespace emu

#define __syncthreads() emu::sync()

inline float atomicAdd(float* addr, float v) { float old = *addr; *addr = old + v; return old; }
inline unsigned int atomicAdd(unsigned int* addr, unsigned int v) { unsigned int old = *addr; *addr = old + v; return old; }

// ---- runtime API ----
typedef int cudaError_t;
typedef void* cudaStream_t;
enum { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                      cudaMemcpyDeviceToDevice, cudaMemcpyDefault };

inline const char* cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "emu error"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

template <typename T> inline cudaError_t cudaMalloc(T** p, size_t bytes) {
    // Memoria nova cheia de bytes 0xFF (NaN): ler algo que nunca foi escrito aparece como NaN.
    void* m = std::malloc(bytes ? bytes : 1);
    if (!m) return cudaErrorMemoryAllocation;
    std::memset(m, 0xFF, bytes);
    *p = static_cast<T*>(m);
    return cudaSuccess;
}
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
