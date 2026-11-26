#ifndef PROCESSAMENTO_H
#define PROCESSAMENTO_H

#include <vector>

// Interface entre o programa principal (main.cpp) e o processamento CUDA.
// Nenhum tipo CUDA aparece aqui: main.cpp e compilado com g++.
// A implementacao destas funcoes fica em processamento.cu.
//
// Convencao dos dados: matrizes guardadas linha a linha em std::vector<float>.
//   entradas: [amostras x camadas.front()]
//   alvos:    [amostras x camadas.back()]

// Funcao de ativacao das camadas ocultas. A camada de saida e sempre linear
// (identidade), o adequado para regressao.
enum class Ativacao { Identidade, Sigmoide, Tanh, ReLU };

enum class Otimizador { SGD, Adam };

// Rede treinada (ou recem-criada). Os parametros ficam na CPU entre as chamadas.
struct ModeloMLP {
    std::vector<int> camadas;       // unidades por camada, ex.: {1, 8, 8, 8, 1}
    Ativacao ativacao;              // ativacao das camadas ocultas
    std::vector<float> parametros;  // [W0 b0 W1 b1 ...]; cada W e [entradas x saidas]
};

struct ConfiguracaoTreino {
    int epocas = 300;                          // passagens completas pelos dados
    int tamanhoLote = 16;                      // amostras por passo do otimizador
    float taxaAprendizado = 0.01f;
    Otimizador otimizador = Otimizador::Adam;
    unsigned semente = 42;                     // ordem aleatoria dos lotes
};

// Cria o contexto CUDA (o primeiro acesso a GPU e lento). Chamar antes de medir tempos.
void inicializarCUDA();

// Cria a rede com pesos aleatorios (inicializacao He/LeCun) e vieses zerados.
ModeloMLP criarMLP(const std::vector<int>& camadas, Ativacao ativacao, unsigned semente);

// Treina o modelo na GPU (MSE). Devolve a perda media de cada epoca.
std::vector<float> treinarMLP(ModeloMLP& modelo,
                              const std::vector<float>& entradas,
                              const std::vector<float>& alvos,
                              const ConfiguracaoTreino& config);

// Calcula as saidas da rede para cada amostra: [amostras x camadas.back()].
std::vector<float> preverMLP(const ModeloMLP& modelo,
                             const std::vector<float>& entradas);

#endif
