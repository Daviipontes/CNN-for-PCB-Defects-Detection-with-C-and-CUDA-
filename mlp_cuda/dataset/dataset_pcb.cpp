#include "dataset/dataset_pcb.h"

#include "dataset/imagem.h"
#include "dataset/xml_pcb.h"
#include "dataset/zip_reader.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

struct ImagemAnotada {
    int classe;
    std::string xml;    // caminho do XML dentro do zip
    std::string pasta;  // pasta (nome da classe)
};

void acrescentar(Dados& d, const std::vector<std::vector<float>>& recortes, int classe) {
    for (const auto& r : recortes) {
        d.entradas.insert(d.entradas.end(), r.begin(), r.end());
        d.rotulos.push_back(classe);
    }
    d.amostras = static_cast<int>(d.rotulos.size());
}

// Embaralha as amostras (para nao ficarem agrupadas por classe) e monta os alvos one-hot.
void finalizar(Dados& d, int dimensao, std::mt19937& gerador) {
    std::vector<int> ordem(d.amostras);
    std::iota(ordem.begin(), ordem.end(), 0);
    std::shuffle(ordem.begin(), ordem.end(), gerador);

    Dados f;
    f.amostras = d.amostras;
    f.dimensao = dimensao;
    f.classes = d.classes;
    f.entradas.resize(d.entradas.size());
    f.alvos.assign(static_cast<size_t>(d.amostras) * d.classes, 0.0f);
    f.rotulos.resize(d.amostras);
    for (int i = 0; i < d.amostras; ++i) {
        const int origem = ordem[i];
        std::copy_n(&d.entradas[static_cast<size_t>(origem) * dimensao], dimensao,
                    &f.entradas[static_cast<size_t>(i) * dimensao]);
        f.rotulos[i] = d.rotulos[origem];
        f.alvos[static_cast<size_t>(i) * d.classes + f.rotulos[i]] = 1.0f;
    }
    d = std::move(f);
}

// Raiz do dataset dentro do zip: tudo que vem antes de "Annotations/".
std::string acharRaiz(const std::vector<std::string>& nomes) {
    for (const std::string& n : nomes) {
        const size_t p = n.find("Annotations/");
        if (p != std::string::npos && (p == 0 || n[p - 1] == '/')) return n.substr(0, p);
    }
    throw std::runtime_error("Nao encontrei Annotations/ no zip");
}

// Lista os XML de Annotations/<classe>/ e devolve os nomes das classes (ordenados).
std::vector<ImagemAnotada> listarAnotacoes(const std::vector<std::string>& nomes, const std::string& raiz,
                                           std::vector<std::string>& classes) {
    std::vector<ImagemAnotada> imagens;
    const std::string prefixo = raiz + "Annotations/";
    for (const std::string& n : nomes) {
        if (n.compare(0, prefixo.size(), prefixo) != 0 || n.size() < 4 || n.compare(n.size() - 4, 4, ".xml") != 0)
            continue;
        const size_t barra = n.find('/', prefixo.size());
        if (barra == std::string::npos) continue;
        const std::string pasta = n.substr(prefixo.size(), barra - prefixo.size());
        if (std::find(classes.begin(), classes.end(), pasta) == classes.end()) classes.push_back(pasta);
        imagens.push_back({0, n, pasta});
    }
    std::sort(classes.begin(), classes.end());
    for (ImagemAnotada& im : imagens)
        im.classe = static_cast<int>(std::find(classes.begin(), classes.end(), im.pasta) - classes.begin());
    return imagens;
}

}  // namespace

DatasetPCB carregarPCB(const std::string& caminhoZip, int lado, float fracaoTeste, unsigned semente) {
    if (lado < 1) throw std::runtime_error("O lado do recorte deve ser pelo menos 1");
    LeitorZip zip(caminhoZip);

    const std::vector<std::string> nomes = zip.nomes();
    const std::string raiz = acharRaiz(nomes);

    DatasetPCB saida;
    const std::vector<ImagemAnotada> imagens = listarAnotacoes(nomes, raiz, saida.classes);
    const int C = static_cast<int>(saida.classes.size());
    if (C < 2) throw std::runtime_error("O dataset precisa de pelo menos 2 classes");
    const int dimensao = lado * lado;
    std::mt19937 gerador(semente);

    // Divisao por imagem, estratificada por classe.
    Dados treino, teste;
    treino.classes = teste.classes = C;
    for (int c = 0; c < C; ++c) {
        std::vector<ImagemAnotada> daClasse;
        for (const ImagemAnotada& im : imagens)
            if (im.classe == c) daClasse.push_back(im);
        std::sort(daClasse.begin(), daClasse.end(),
                  [](const ImagemAnotada& a, const ImagemAnotada& b) { return a.xml < b.xml; });
        std::shuffle(daClasse.begin(), daClasse.end(), gerador);
        const size_t nTeste = std::max<size_t>(1, static_cast<size_t>(std::lround(daClasse.size() * fracaoTeste)));

        for (size_t k = 0; k < daClasse.size(); ++k) {
            const std::vector<std::uint8_t> bruto = zip.ler(daClasse[k].xml);
            const Anotacao nota = lerAnotacao(std::string(bruto.begin(), bruto.end()));
            const std::string caminho = raiz + "images/" + saida.classes[c] + "/" + nota.arquivoImagem;
            acrescentar(k < nTeste ? teste : treino,
                        recortesDaImagem(zip.ler(caminho), nota.caixas, lado, caminho), c);
        }
        std::cout << "  " << saida.classes[c] << ": " << daClasse.size() << " imagens\n";
    }

    finalizar(treino, dimensao, gerador);
    finalizar(teste, dimensao, gerador);
    if (treino.amostras == 0 || teste.amostras == 0)
        throw std::runtime_error("Dataset sem amostras suficientes para treino e teste");
    saida.treino = std::move(treino);
    saida.teste = std::move(teste);
    return saida;
}
