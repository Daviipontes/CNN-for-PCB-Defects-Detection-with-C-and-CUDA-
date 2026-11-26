#ifndef DATASET_DADOS_H
#define DATASET_DADOS_H

#include <vector>

// Conjunto de amostras para classificacao, pronto para a rede.
struct Dados {
    int amostras = 0, dimensao = 0, classes = 0;
    std::vector<float> entradas;  // [amostras x dimensao]
    std::vector<float> alvos;     // [amostras x classes], one-hot
    std::vector<int> rotulos;     // classe de cada amostra
};

#endif
