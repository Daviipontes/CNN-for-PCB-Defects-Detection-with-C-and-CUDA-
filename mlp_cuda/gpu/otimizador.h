#ifndef GPU_OTIMIZADOR_H
#define GPU_OTIMIZADOR_H

#include "processamento.h"

// Um passo do otimizador na GPU sobre os 'n' parametros (todos num unico buffer continuo, entao
// um lancamento atualiza a rede inteira). 'p' = parametros, 'g' = gradientes, 'm' e 'v' = estado
// do otimizador (SGD usa so 'm' como velocidade). 'passo' comeca em 1 (correcao de vies do Adam).
// Lanca std::runtime_error se o lancamento do kernel falhar.
void passoOtimizador(Otimizador tipo, float* p, float* m, float* v, const float* g,
                     int n, float taxa, int passo);

#endif
