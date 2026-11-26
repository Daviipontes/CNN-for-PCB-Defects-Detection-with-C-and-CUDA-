#ifndef DATASET_ZIP_READER_H
#define DATASET_ZIP_READER_H

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// Le entradas de um .zip direto do disco (o archive.zip tem ~2 GB: nada e carregado por inteiro).
// Suporta os metodos "armazenado" e "deflate" e o campo extra zip64 das entradas.
class LeitorZip {
public:
    // Abre o zip e le o diretorio central. Lanca std::runtime_error se o arquivo for invalido.
    explicit LeitorZip(const std::string& caminho);

    // Caminhos de todos os arquivos do zip (diretorios nao entram).
    std::vector<std::string> nomes() const;

    // Descomprime uma entrada na memoria.
    std::vector<std::uint8_t> ler(const std::string& nome);

private:
    struct Entrada {
        std::uint16_t metodo;
        std::uint64_t tamComprimido, tamOriginal, deslocamento;
    };
    std::ifstream arquivo;
    std::map<std::string, Entrada> entradas;

    void lerBytes(std::uint64_t posicao, void* destino, size_t n);
    void lerDiretorioCentral();
};

#endif
