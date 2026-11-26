#include "avaliacao.h"
#include "dataset/dataset_pcb.h"
#include "processamento.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Converte um argumento da linha de comando em inteiro, com mensagem de erro legivel.
static int lerInteiro(const char* texto, const char* nome) {
    try {
        size_t lidos = 0;
        const int valor = std::stoi(texto, &lidos);
        if (lidos == std::string(texto).size()) return valor;
    } catch (const std::exception&) {}
    throw std::runtime_error(std::string(nome) + " deve ser um numero inteiro: " + texto);
}

// Programa principal: carrega o dataset de defeitos de PCB, treina o MLP na GPU e avalia.
// Uso: ./mlp_cuda [arquivo_zip] [epocas] [tamanho_lote]
// O zip (padrao: ../archive.zip) e lido direto, sem extrair; ver dataset/dataset_pcb.h.
// A saida tem nome FIXO: saida.csv (previsoes do conjunto de teste), no diretorio de execucao.
int main(int argc, char** argv) {
    if (argc > 4) {
        std::cerr << "Uso: " << argv[0] << " [arquivo_zip] [epocas] [tamanho_lote]\n";
        return 1;
    }

    try {
        // Parametros opcionais (padrao: 100 epocas, lotes de 32 amostras).
        const std::string caminhoZip = argc >= 2 ? argv[1] : "../archive.zip";
        ConfiguracaoTreino config;
        config.epocas = 100;
        config.tamanhoLote = 32;
        config.taxaAprendizado = 0.001f;
        if (argc >= 3) config.epocas = lerInteiro(argv[2], "epocas");
        if (argc == 4) config.tamanhoLote = lerInteiro(argv[3], "tamanho_lote");
        if (config.epocas < 1) throw std::runtime_error("O numero de epocas deve ser pelo menos 1");
        if (config.tamanhoLote < 1) throw std::runtime_error("O lote deve ter pelo menos 1 amostra");

        std::cout << "Lendo " << caminhoZip << " ...\n";
        const DatasetPCB dataset = carregarPCB(caminhoZip);
        const Dados& treino = dataset.treino;
        const Dados& teste = dataset.teste;
        std::cout << "Treino: " << treino.amostras << " amostras | Teste: " << teste.amostras
                  << " amostras | Entradas: " << treino.dimensao << " | Classes: " << treino.classes << '\n';

        // Arquitetura: entrada (pixels), duas camadas ocultas (tanh), uma saida linear por classe.
        // Cada saida e treinada para 1 na classe correta e 0 nas demais (MSE sobre one-hot).
        ModeloMLP modelo = criarMLP({treino.dimensao, 128, 64, treino.classes}, Ativacao::Tanh, 42);

        // Cria o contexto CUDA fora da medicao de tempo.
        inicializarCUDA();
        const auto inicio = std::chrono::steady_clock::now();

        // Treina TODA a rede na GPU em uma chamada.
        const std::vector<float> perdas = treinarMLP(modelo, treino.entradas, treino.alvos, config);

        const double segundos = std::chrono::duration<double>(std::chrono::steady_clock::now() - inicio).count();

        // Mostra a perda em 10 pontos do treino.
        const int intervalo = config.epocas >= 10 ? config.epocas / 10 : 1;
        for (int e = intervalo; e <= config.epocas; e += intervalo)
            std::cout << "Epoca " << e << " - perda (MSE): " << perdas[e - 1] << '\n';

        // Avaliacao nos dois conjuntos.
        const std::vector<int> previstoTreino = classesPrevistas(preverMLP(modelo, treino.entradas), treino.classes);
        const std::vector<int> previstoTeste = classesPrevistas(preverMLP(modelo, teste.entradas), teste.classes);

        std::cout << "\nMatriz de confusao (teste; linhas = real, colunas = previsto):\n";
        imprimirMatrizConfusao(std::cout, matrizConfusao(previstoTeste, teste.rotulos, teste.classes), dataset.classes);

        gravarPrevisoes("saida.csv", previstoTeste, teste.rotulos, dataset.classes);

        std::cout << "\nArquivo gerado: saida.csv\n"
                  << "Epocas: " << config.epocas << " | Lote: " << config.tamanhoLote << '\n'
                  << std::fixed << std::setprecision(2)
                  << "Acuracia no treino: " << 100.0 * acuracia(previstoTreino, treino.rotulos) << "%\n"
                  << "Acuracia no teste: " << 100.0 * acuracia(previstoTeste, teste.rotulos) << "%\n"
                  << std::defaultfloat
                  << "Tempo de treino: " << segundos << " s\n"
                  << "Taxa media: " << (segundos > 0 ? config.epocas * treino.amostras / segundos : 0)
                  << " amostras/s\n";
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
