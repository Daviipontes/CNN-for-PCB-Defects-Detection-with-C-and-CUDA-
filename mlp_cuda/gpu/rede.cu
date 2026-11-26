#include "gpu/rede.h"

#include "gpu/gemm.h"

#include <stdexcept>
#include <string>

Deslocamentos calcularDeslocamentos(const std::vector<int>& camadas) {
    Deslocamentos d;
    for (size_t l = 0; l + 1 < camadas.size(); ++l) {
        d.W.push_back(d.total); d.total += (size_t)camadas[l] * camadas[l + 1];
        d.b.push_back(d.total); d.total += (size_t)camadas[l + 1];
    }
    return d;
}

AtivacoesGPU::AtivacoesGPU(const std::vector<int>& camadas, int lote) {
    for (int unidades : camadas)
        ativacoes.emplace_back((size_t)lote * unidades, "cudaMalloc ativacoes");
}

void propagar(const std::vector<int>& camadas, Ativacao ativacao, const Deslocamentos& d,
              const float* parametros, AtivacoesGPU& rede, int linhas) {
    const size_t L = camadas.size() - 1;
    for (size_t l = 0; l < L; ++l) {
        const Ativacao a = (l + 1 == L) ? Ativacao::Identidade : ativacao;
        gemm(rede.ativacoes[l].get(), parametros + d.W[l], rede.ativacoes[l + 1].get(),
             linhas, camadas[l + 1], camadas[l], false, false,
             Epilogo::ViesAtivacao, parametros + d.b[l], nullptr, a);
    }
}

void validarModelo(const ModeloMLP& modelo) {
    if (modelo.camadas.size() < 2)
        throw std::runtime_error("A rede precisa de pelo menos 2 camadas");
    for (int unidades : modelo.camadas)
        if (unidades < 1) throw std::runtime_error("Toda camada precisa de pelo menos 1 unidade");
    if (modelo.parametros.size() != calcularDeslocamentos(modelo.camadas).total)
        throw std::runtime_error("Numero de parametros nao corresponde as camadas");
}

int contarAmostras(const std::vector<float>& dados, int porAmostra, const char* nome) {
    if (dados.empty() || dados.size() % porAmostra != 0)
        throw std::runtime_error(std::string(nome) + ": tamanho nao e multiplo do numero de unidades da camada");
    return static_cast<int>(dados.size() / porAmostra);
}
