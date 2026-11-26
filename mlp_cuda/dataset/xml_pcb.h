#ifndef DATASET_XML_PCB_H
#define DATASET_XML_PCB_H

#include <string>
#include <vector>

// Caixa de um defeito, em pixels da imagem original.
struct Caixa { int xmin, ymin, xmax, ymax; };

// Conteudo util de um XML de anotacao (formato Pascal VOC) do PCB_DATASET.
struct Anotacao {
    std::string arquivoImagem;  // <filename>
    std::vector<Caixa> caixas;  // um <bndbox> por defeito
};

// Le <filename> e todos os <bndbox> do XML. Lanca std::runtime_error se faltar <filename>.
Anotacao lerAnotacao(const std::string& xml);

#endif
