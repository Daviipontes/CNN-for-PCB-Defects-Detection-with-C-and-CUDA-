#include "dataset/imagem.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Decodificador de imagens em um unico cabecalho (dominio publico). So JPEG e PNG, so memoria.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "third_party/stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace {

constexpr float CONTEXTO = 2.0f;  // lado da janela = CONTEXTO * maior lado da caixa
constexpr int LADO_MINIMO = 64;   // em pixels da imagem original

// Reduz o recorte quadrado [x0, x0+l) x [y0, y0+l) para lado x lado pela media dos pixels de
// cada bloco e padroniza o resultado (media 0, desvio 1: remove diferencas de brilho).
void recortar(const std::uint8_t* img, int largura, int x0, int y0, int l, int lado, float* saida) {
    for (int oy = 0; oy < lado; ++oy) {
        const int ya = y0 + oy * l / lado;
        const int yb = std::max(ya + 1, y0 + (oy + 1) * l / lado);
        for (int ox = 0; ox < lado; ++ox) {
            const int xa = x0 + ox * l / lado;
            const int xb = std::max(xa + 1, x0 + (ox + 1) * l / lado);
            double soma = 0.0;
            for (int y = ya; y < yb; ++y)
                for (int x = xa; x < xb; ++x) soma += img[static_cast<size_t>(y) * largura + x];
            saida[oy * lado + ox] = static_cast<float>(soma / ((yb - ya) * (xb - xa)));
        }
    }
    double s = 0.0, s2 = 0.0;
    const int n = lado * lado;
    for (int i = 0; i < n; ++i) { s += saida[i]; s2 += double(saida[i]) * saida[i]; }
    const double media = s / n;
    const double desvio = std::sqrt(std::max(s2 / n - media * media, 1e-6));
    for (int i = 0; i < n; ++i) saida[i] = static_cast<float>((saida[i] - media) / desvio);
}

}  // namespace

std::vector<std::vector<float>> recortesDaImagem(const std::vector<std::uint8_t>& arquivo,
                                                 const std::vector<Caixa>& caixas, int lado,
                                                 const std::string& nome) {
    int largura = 0, altura = 0, canais = 0;
    std::uint8_t* img = stbi_load_from_memory(arquivo.data(), static_cast<int>(arquivo.size()),
                                              &largura, &altura, &canais, 1);
    if (!img) throw std::runtime_error("Nao foi possivel decodificar a imagem " + nome);

    std::vector<std::vector<float>> recortes;
    for (const Caixa& c : caixas) {
        int l = static_cast<int>(std::max<float>(CONTEXTO * std::max(c.xmax - c.xmin, c.ymax - c.ymin), LADO_MINIMO));
        l = std::min({l, largura, altura});
        // Janela centrada na caixa, deslocada para ficar dentro da imagem.
        const int x0 = std::clamp((c.xmin + c.xmax) / 2 - l / 2, 0, largura - l);
        const int y0 = std::clamp((c.ymin + c.ymax) / 2 - l / 2, 0, altura - l);
        recortes.emplace_back(static_cast<size_t>(lado) * lado);
        recortar(img, largura, x0, y0, l, lado, recortes.back().data());
    }
    stbi_image_free(img);
    return recortes;
}
