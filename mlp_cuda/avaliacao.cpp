#include "avaliacao.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>

std::vector<int> classesPrevistas(const std::vector<float>& saidas, int classes) {
    std::vector<int> previstas(saidas.size() / classes);
    for (size_t i = 0; i < previstas.size(); ++i) {
        const float* s = &saidas[i * classes];
        previstas[i] = static_cast<int>(std::max_element(s, s + classes) - s);
    }
    return previstas;
}

double acuracia(const std::vector<int>& previstas, const std::vector<int>& reais) {
    if (reais.empty()) return 0.0;
    int acertos = 0;
    for (size_t i = 0; i < reais.size(); ++i) acertos += previstas[i] == reais[i];
    return static_cast<double>(acertos) / reais.size();
}

std::vector<int> matrizConfusao(const std::vector<int>& previstas, const std::vector<int>& reais, int classes) {
    std::vector<int> confusao(static_cast<size_t>(classes) * classes, 0);
    for (size_t i = 0; i < reais.size(); ++i)
        ++confusao[static_cast<size_t>(reais[i]) * classes + previstas[i]];
    return confusao;
}

void imprimirMatrizConfusao(std::ostream& saida, const std::vector<int>& confusao,
                            const std::vector<std::string>& nomes) {
    const int C = static_cast<int>(nomes.size());
    const auto formatoAnterior = saida.flags();
    const auto precisaoAnterior = saida.precision();
    for (int r = 0; r < C; ++r) {
        int total = 0;
        saida << std::left << std::setw(16) << nomes[r] << std::right;
        for (int c = 0; c < C; ++c) {
            saida << std::setw(5) << confusao[static_cast<size_t>(r) * C + c];
            total += confusao[static_cast<size_t>(r) * C + c];
        }
        const double acerto = total ? 100.0 * confusao[static_cast<size_t>(r) * C + r] / total : 0.0;
        saida << "   " << std::fixed << std::setprecision(1) << acerto << "%\n";
    }
    saida.flags(formatoAnterior);
    saida.precision(precisaoAnterior);
}

void gravarPrevisoes(const std::string& caminho, const std::vector<int>& previstas,
                     const std::vector<int>& reais, const std::vector<std::string>& nomes) {
    std::ofstream arquivo(caminho);
    if (!arquivo) throw std::runtime_error("Nao foi possivel criar " + caminho);
    arquivo << "indice,classe_real,classe_prevista\n";
    for (size_t i = 0; i < reais.size(); ++i)
        arquivo << i << ',' << nomes[reais[i]] << ',' << nomes[previstas[i]] << '\n';
}
