#include "dataset/xml_pcb.h"

#include <cstdlib>
#include <stdexcept>

namespace {

// Valor do texto entre <tag> e </tag>, procurando a partir de 'pos'. Atualiza 'pos' para depois da tag.
bool valorTag(const std::string& xml, const char* tag, size_t& pos, std::string& valor) {
    const std::string abre = std::string("<") + tag + ">", fecha = std::string("</") + tag + ">";
    const size_t i = xml.find(abre, pos);
    if (i == std::string::npos) return false;
    const size_t j = xml.find(fecha, i);
    if (j == std::string::npos) return false;
    valor = xml.substr(i + abre.size(), j - i - abre.size());
    pos = j + fecha.size();
    return true;
}

}  // namespace

Anotacao lerAnotacao(const std::string& xml) {
    Anotacao a;
    size_t pos = 0;
    if (!valorTag(xml, "filename", pos, a.arquivoImagem))
        throw std::runtime_error("XML sem <filename>");

    pos = 0;
    std::string bloco;
    while (valorTag(xml, "bndbox", pos, bloco)) {
        size_t q = 0;
        std::string v[4];
        if (valorTag(bloco, "xmin", q, v[0]) && valorTag(bloco, "ymin", q, v[1]) &&
            valorTag(bloco, "xmax", q, v[2]) && valorTag(bloco, "ymax", q, v[3]))
            a.caixas.push_back({std::atoi(v[0].c_str()), std::atoi(v[1].c_str()),
                                std::atoi(v[2].c_str()), std::atoi(v[3].c_str())});
    }
    return a;
}
