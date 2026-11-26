#include "processamento.h"

#include "gpu/cuda_util.h"
#include "gpu/gemm.h"
#include "gpu/kernels_aux.h"
#include "gpu/otimizador.h"
#include "gpu/rede.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// Interface publica do processamento (ver processamento.h): so orquestra o treino.
// Os detalhes da GPU ficam em gpu/:
//  - gemm: produto de matrizes com memoria compartilhada e registradores (tiles), com vies,
//    ativacao e derivada fundidos no mesmo kernel;
//  - kernels_aux: soma por coluna, perda MSE e montagem do lote;
//  - otimizador: SGD/Adam, um unico lancamento atualiza todos os parametros;
//  - rede: layout dos parametros, buffers de ativacao e propagacao.
// Pesos, ativacoes, gradientes e o conjunto de dados ficam na GPU durante TODO o treino.

// ==================== FUNCOES DA INTERFACE ====================

void inicializarCUDA() {
    verificarCUDA(cudaFree(0), "Inicializacao do CUDA");
}

ModeloMLP criarMLP(const std::vector<int>& camadas, Ativacao ativacao, unsigned semente) {
    ModeloMLP modelo{camadas, ativacao, {}};
    if (camadas.size() < 2) throw std::runtime_error("A rede precisa de pelo menos 2 camadas");
    const Deslocamentos d = calcularDeslocamentos(camadas);
    modelo.parametros.assign(d.total, 0.0f);  // vieses comecam em zero

    // He para ReLU (desvio = sqrt(2 / entradas)); LeCun para tanh/sigmoide (sqrt(1 / entradas)).
    std::mt19937 gerador(semente);
    const float ganho = (ativacao == Ativacao::ReLU) ? 2.0f : 1.0f;
    for (size_t l = 0; l + 1 < camadas.size(); ++l) {
        std::normal_distribution<float> normal(0.0f, std::sqrt(ganho / camadas[l]));
        for (size_t i = 0; i < (size_t)camadas[l] * camadas[l + 1]; ++i)
            modelo.parametros[d.W[l] + i] = normal(gerador);
    }
    return modelo;
}

std::vector<float> treinarMLP(ModeloMLP& modelo,
                              const std::vector<float>& entradas,
                              const std::vector<float>& alvos,
                              const ConfiguracaoTreino& config) {
    validarModelo(modelo);
    if (config.epocas < 1 || config.tamanhoLote < 1 || config.taxaAprendizado <= 0.0f)
        throw std::runtime_error("Configuracao de treino invalida");

    const std::vector<int>& camadas = modelo.camadas;
    const int nEntrada = camadas.front();
    const int nSaida = camadas.back();
    const size_t L = camadas.size() - 1;
    const int amostras = contarAmostras(entradas, nEntrada, "entradas");
    if (contarAmostras(alvos, nSaida, "alvos") != amostras)
        throw std::runtime_error("Entradas e alvos tem quantidades diferentes de amostras");

    const Deslocamentos d = calcularDeslocamentos(camadas);
    const int nParametros = static_cast<int>(d.total);
    const int lote = std::min(config.tamanhoLote, amostras);

    // Reserva TODA a memoria do treino uma unica vez (prefixo d_ = device).
    MemoriaGPU<float> d_parametros(d.total, "cudaMalloc parametros");
    MemoriaGPU<float> d_gradientes(d.total, "cudaMalloc gradientes");
    MemoriaGPU<float> d_m(d.total, "cudaMalloc estado do otimizador (m)");
    MemoriaGPU<float> d_v(d.total, "cudaMalloc estado do otimizador (v)");
    MemoriaGPU<float> d_entradas(entradas.size(), "cudaMalloc entradas");
    MemoriaGPU<float> d_alvos(alvos.size(), "cudaMalloc alvos");
    MemoriaGPU<float> d_alvosLote((size_t)lote * nSaida, "cudaMalloc alvos do lote");
    MemoriaGPU<int> d_ordem(amostras, "cudaMalloc ordem das amostras");
    MemoriaGPU<float> d_somaErros(1, "cudaMalloc soma dos erros");
    AtivacoesGPU rede(camadas, lote);
    // deltas[l] = dPerda/dZ da camada l (mesmo formato de ativacoes[l]); deltas[0] nao e usado.
    std::vector<MemoriaGPU<float>> deltas;
    for (size_t l = 0; l <= L; ++l)
        deltas.emplace_back((size_t)lote * (l == 0 ? 1 : camadas[l]), "cudaMalloc deltas");

    // Copia parametros e o conjunto de dados INTEIRO para a GPU, uma unica vez.
    verificarCUDA(cudaMemcpy(d_parametros.get(), modelo.parametros.data(), d.total * sizeof(float),
                             cudaMemcpyHostToDevice), "cudaMemcpy parametros CPU -> GPU");
    verificarCUDA(cudaMemset(d_m.get(), 0, d.total * sizeof(float)), "cudaMemset m");
    verificarCUDA(cudaMemset(d_v.get(), 0, d.total * sizeof(float)), "cudaMemset v");
    verificarCUDA(cudaMemcpy(d_entradas.get(), entradas.data(), entradas.size() * sizeof(float),
                             cudaMemcpyHostToDevice), "cudaMemcpy entradas CPU -> GPU");
    verificarCUDA(cudaMemcpy(d_alvos.get(), alvos.data(), alvos.size() * sizeof(float),
                             cudaMemcpyHostToDevice), "cudaMemcpy alvos CPU -> GPU");

    float* P = d_parametros.get();
    float* G = d_gradientes.get();

    std::vector<int> ordem(amostras);
    std::iota(ordem.begin(), ordem.end(), 0);
    std::mt19937 gerador(config.semente);
    std::vector<float> historico;
    int passo = 0;

    for (int epoca = 0; epoca < config.epocas; ++epoca) {

        // Nova ordem aleatoria a cada epoca: so os indices viajam para a GPU.
        std::shuffle(ordem.begin(), ordem.end(), gerador);
        verificarCUDA(cudaMemcpy(d_ordem.get(), ordem.data(), amostras * sizeof(int),
                                 cudaMemcpyHostToDevice), "cudaMemcpy ordem CPU -> GPU");
        verificarCUDA(cudaMemset(d_somaErros.get(), 0, sizeof(float)), "cudaMemset soma dos erros");

        // O ultimo lote pode ter menos amostras que os anteriores.
        for (int inicio = 0; inicio < amostras; inicio += lote) {
            const int linhas = std::min(lote, amostras - inicio);

            // 1) Monta o lote na GPU a partir dos indices embaralhados.
            reunirLinhas(d_entradas.get(), d_ordem.get() + inicio, rede.ativacoes[0].get(), linhas, nEntrada);
            reunirLinhas(d_alvos.get(), d_ordem.get() + inicio, d_alvosLote.get(), linhas, nSaida);

            // 2) Propagacao.
            propagar(camadas, modelo.ativacao, d, P, rede, linhas);

            // 3) Perda e gradiente da saida: delta[L] = 2 (previsto - alvo) / n.
            perdaMSE(rede.ativacoes[L].get(), d_alvosLote.get(), deltas[L].get(), linhas * nSaida,
                     d_somaErros.get());

            // 4) Retropropagacao (regra da cadeia), da ultima camada para a primeira:
            //    dW[l]    = A[l]^T * delta[l+1]                 (GEMM com A transposta)
            //    db[l]    = soma no lote de delta[l+1]          (soma por coluna)
            //    delta[l] = (delta[l+1] * W[l]^T) * f'(A[l])    (GEMM com W transposta + derivada)
            //    Nenhum peso muda aqui: todas as camadas usam os pesos de antes da atualizacao.
            for (size_t l = L; l-- > 0;) {
                const int nIn = camadas[l], nOut = camadas[l + 1];
                gemm(rede.ativacoes[l].get(), deltas[l + 1].get(), G + d.W[l], nIn, nOut, linhas,
                     true, false, Epilogo::Nenhum, nullptr, nullptr, Ativacao::Identidade);
                somaColunas(deltas[l + 1].get(), G + d.b[l], linhas, nOut);
                if (l > 0)
                    gemm(deltas[l + 1].get(), P + d.W[l], deltas[l].get(), linhas, nIn, nOut,
                         false, true, Epilogo::GradienteAtivacao, nullptr, rede.ativacoes[l].get(),
                         modelo.ativacao);
            }

            // 5) Um lancamento atualiza todos os parametros.
            ++passo;
            passoOtimizador(config.otimizador, P, d_m.get(), d_v.get(), G, nParametros,
                            config.taxaAprendizado, passo);
        }

        // Unico ponto em que a CPU espera a GPU durante a epoca: leitura da perda.
        float somaErros = 0.0f;
        verificarCUDA(cudaMemcpy(&somaErros, d_somaErros.get(), sizeof(float),
                                 cudaMemcpyDeviceToHost), "cudaMemcpy perda GPU -> CPU");
        historico.push_back(somaErros / (static_cast<float>(amostras) * nSaida));
    }

    // Devolve os parametros treinados para a CPU.
    verificarCUDA(cudaMemcpy(modelo.parametros.data(), P, d.total * sizeof(float),
                             cudaMemcpyDeviceToHost), "cudaMemcpy parametros GPU -> CPU");
    return historico;
}

std::vector<float> preverMLP(const ModeloMLP& modelo, const std::vector<float>& entradas) {
    validarModelo(modelo);
    const std::vector<int>& camadas = modelo.camadas;
    const int nEntrada = camadas.front();
    const int nSaida = camadas.back();
    const int amostras = contarAmostras(entradas, nEntrada, "entradas");
    const Deslocamentos d = calcularDeslocamentos(camadas);

    // Processa em pedacos de ate 1024 amostras para limitar a memoria das ativacoes.
    const int pedaco = std::min(amostras, 1024);
    MemoriaGPU<float> d_parametros(d.total, "cudaMalloc parametros");
    AtivacoesGPU rede(camadas, pedaco);
    verificarCUDA(cudaMemcpy(d_parametros.get(), modelo.parametros.data(), d.total * sizeof(float),
                             cudaMemcpyHostToDevice), "cudaMemcpy parametros CPU -> GPU");

    std::vector<float> saidas((size_t)amostras * nSaida);
    for (int inicio = 0; inicio < amostras; inicio += pedaco) {
        const int linhas = std::min(pedaco, amostras - inicio);
        verificarCUDA(cudaMemcpy(rede.ativacoes[0].get(), entradas.data() + (size_t)inicio * nEntrada,
                                 (size_t)linhas * nEntrada * sizeof(float), cudaMemcpyHostToDevice),
                      "cudaMemcpy entradas CPU -> GPU");
        propagar(camadas, modelo.ativacao, d, d_parametros.get(), rede, linhas);
        verificarCUDA(cudaMemcpy(saidas.data() + (size_t)inicio * nSaida, rede.ativacoes.back().get(),
                                 (size_t)linhas * nSaida * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy saidas GPU -> CPU");
    }
    return saidas;
}
