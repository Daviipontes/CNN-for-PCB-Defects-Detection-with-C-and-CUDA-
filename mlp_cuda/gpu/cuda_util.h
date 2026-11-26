#ifndef GPU_CUDA_UTIL_H
#define GPU_CUDA_UTIL_H

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// Converte erros da API CUDA em excecoes com mensagem legivel.
inline void verificarCUDA(cudaError_t erro, const char* etapa) {
    if (erro != cudaSuccess)
        throw std::runtime_error(std::string(etapa) + ": " + cudaGetErrorString(erro));
}

// Buffer na memoria da GPU que se libera sozinho ao sair de escopo (inclusive em caso de
// excecao). Faz o papel do try/catch com cudaFree, util quando ha muitos buffers.
template <typename T>
class MemoriaGPU {
public:
    MemoriaGPU(size_t quantidade, const char* etapa) {
        verificarCUDA(cudaMalloc(&ptr, quantidade * sizeof(T)), etapa);
    }
    ~MemoriaGPU() { if (ptr) cudaFree(ptr); }
    MemoriaGPU(MemoriaGPU&& outro) noexcept : ptr(outro.ptr) { outro.ptr = nullptr; }
    MemoriaGPU(const MemoriaGPU&) = delete;
    MemoriaGPU& operator=(const MemoriaGPU&) = delete;
    T* get() const { return ptr; }
private:
    T* ptr = nullptr;
};

#endif
