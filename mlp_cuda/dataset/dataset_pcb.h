#ifndef DATASET_DATASET_PCB_H
#define DATASET_DATASET_PCB_H

#include "dataset/dados.h"

#include <string>
#include <vector>

// Monta o dataset de defeitos de PCB direto do archive.zip (sem extrair nada em disco).
//
// Cada defeito anotado nos XML (<bndbox>) vira uma amostra (ver recortesDaImagem). A classe e a
// pasta do defeito em Annotations/. A divisao treino/teste e feita POR IMAGEM, estratificada por
// classe, para que defeitos da mesma placa nao aparecam nos dois conjuntos.
struct DatasetPCB {
    Dados treino, teste;
    std::vector<std::string> classes;  // nome de cada classe (indice = rotulo)
};

DatasetPCB carregarPCB(const std::string& caminhoZip, int lado = 32,
                       float fracaoTeste = 0.2f, unsigned semente = 42);

#endif
