#ifndef AVALIACAO_H
#define AVALIACAO_H

#include "dataset/dados.h"

#include <iosfwd>
#include <string>
#include <vector>

// Metricas de classificacao, independentes da GPU.

// Classe prevista de cada amostra = indice da maior saida da rede. saidas: [amostras x classes].
std::vector<int> classesPrevistas(const std::vector<float>& saidas, int classes);

// Fracao de acertos (0 a 1).
double acuracia(const std::vector<int>& previstas, const std::vector<int>& reais);

// Matriz de confusao [classes x classes]: linha = classe real, coluna = classe prevista.
std::vector<int> matrizConfusao(const std::vector<int>& previstas, const std::vector<int>& reais, int classes);

// Imprime a matriz com o nome de cada classe e o acerto por classe (recall).
void imprimirMatrizConfusao(std::ostream& saida, const std::vector<int>& confusao,
                            const std::vector<std::string>& nomes);

// Grava "indice,classe_real,classe_prevista" para cada amostra. Lanca std::runtime_error se falhar.
void gravarPrevisoes(const std::string& caminho, const std::vector<int>& previstas,
                     const std::vector<int>& reais, const std::vector<std::string>& nomes);

#endif
