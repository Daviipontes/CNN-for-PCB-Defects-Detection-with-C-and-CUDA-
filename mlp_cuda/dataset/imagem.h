#ifndef DATASET_IMAGEM_H
#define DATASET_IMAGEM_H

#include "dataset/xml_pcb.h"

#include <cstdint>
#include <string>
#include <vector>

// Decodifica uma imagem (JPEG/PNG) na memoria, em tons de cinza, e devolve um recorte por caixa.
// Cada recorte e uma janela quadrada centrada na caixa (com o dobro do tamanho dela, para dar
// contexto), reduzida para lado x lado pixels e padronizada (media 0, desvio 1).
// 'nome' so aparece na mensagem de erro. Lanca std::runtime_error se nao decodificar.
std::vector<std::vector<float>> recortesDaImagem(const std::vector<std::uint8_t>& arquivo,
                                                 const std::vector<Caixa>& caixas, int lado,
                                                 const std::string& nome);

#endif
