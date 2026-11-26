// Testes emulados na CPU:
//  1. cada kernel comparado com uma versao simples em double;
//  2. a implementacao de referencia (double) conferida por diferencas finitas;
//  3. treinarMLP acompanhando a referencia passo a passo (mesma inicializacao, mesma ordem dos lotes,
//     SGD e Adam, lote final incompleto);
//  4. preverMLP, validacoes de entrada e convergencia em sin^2.
// Os modulos .cu sao compilados como C++ por executar_testes.py (com a emulacao de CUDA) e
// ligados a este teste; aqui so se incluem os cabecalhos, como em qualquer outro cliente.
#include "gpu/cuda_util.h"
#include "gpu/gemm.h"
#include "gpu/kernels_aux.h"
#include "gpu/otimizador.h"
#include "gpu/rede.h"
#include "processamento.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static int falhas = 0;
#define VERIFICAR(cond, ...)                                                                      \
    do {                                                                                          \
        if (!(cond)) { ++falhas; std::printf("  FALHOU: " __VA_ARGS__); std::printf("\n"); }     \
        else { std::printf("  ok: " __VA_ARGS__); std::printf("\n"); }                            \
    } while (0)

static std::mt19937 rng(1234);

static std::vector<float> aleatorio(size_t n, float lo = -1.f, float hi = 1.f) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
static MemoriaGPU<float> paraGPU(const std::vector<float>& h) {
    MemoriaGPU<float> d(h.size(), "teste");
    cudaMemcpy(d.get(), h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}
static std::vector<float> paraCPU(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}
static double f(double z, Ativacao a) {
    switch (a) {
        case Ativacao::Sigmoide: return 1 / (1 + std::exp(-z));
        case Ativacao::Tanh: return std::tanh(z);
        case Ativacao::ReLU: return z > 0 ? z : 0;
        default: return z;
    }
}
static double df(double y, Ativacao a) {
    switch (a) {
        case Ativacao::Sigmoide: return y * (1 - y);
        case Ativacao::Tanh: return 1 - y * y;
        case Ativacao::ReLU: return y > 0 ? 1 : 0;
        default: return 1;
    }
}

// ---------------- kernels ----------------
static void testeGemm() {
    std::printf("GEMM\n");
    const int formas[][3] = {{1, 1, 1}, {1, 64, 1}, {7, 5, 3}, {64, 64, 8}, {65, 63, 9},
                             {100, 1, 100}, {1, 130, 17}, {130, 70, 64}, {33, 129, 1}, {16, 512, 13}};
    const Ativacao ativacoes[] = {Ativacao::Identidade, Ativacao::Sigmoide, Ativacao::Tanh, Ativacao::ReLU};
    int casos = 0, erradas = 0;
    double piorErro = 0;
    for (auto& s : formas) {
        const int M = s[0], N = s[1], K = s[2];
        for (int ta = 0; ta < 2; ++ta)
        for (int tb = 0; tb < 2; ++tb)
        for (int ep = 0; ep < 3; ++ep)
        for (Ativacao at : ativacoes) {
            if (ep == 0 && at != Ativacao::Identidade) continue;
            auto A = aleatorio((size_t)M * K), B = aleatorio((size_t)K * N), vies = aleatorio(N);
            auto aux = aleatorio((size_t)M * N, 0.f, 1.f);
            auto dA = paraGPU(A), dB = paraGPU(B), dv = paraGPU(vies), dx = paraGPU(aux);
            MemoriaGPU<float> dC((size_t)M * N, "teste");
            gemm(dA.get(), dB.get(), dC.get(), M, N, K, ta, tb, (Epilogo)ep, dv.get(), dx.get(), at);
            auto C = paraCPU(dC.get(), (size_t)M * N);
            double erro = 0;
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j) {
                    double s = 0;
                    for (int k = 0; k < K; ++k)
                        s += (ta ? A[(size_t)k * M + i] : A[(size_t)i * K + k]) *
                             (tb ? B[(size_t)j * K + k] : B[(size_t)k * N + j]);
                    if (ep == 1) s = f(s + vies[j], at);
                    if (ep == 2) s *= df(aux[(size_t)i * N + j], at);
                    erro = std::fmax(erro, std::fabs(s - C[(size_t)i * N + j]) / (1 + std::fabs(s)));
                }
            piorErro = std::fmax(piorErro, erro);
            erradas += erro >= 1e-5;
            ++casos;
        }
    }
    VERIFICAR(erradas == 0, "%d casos (formas x transposicoes x epilogos), pior erro relativo %.1e", casos, piorErro);
}

static void testeDemaisKernels() {
    std::printf("soma por coluna, perda MSE, montagem do lote, otimizadores\n");
    {   // soma por coluna
        double erro = 0;
        for (auto rc : std::vector<std::pair<int, int>>{{1, 1}, {1, 33}, {7, 1}, {100, 64}, {9, 65}, {300, 3}}) {
            auto X = aleatorio((size_t)rc.first * rc.second);
            auto dX = paraGPU(X); MemoriaGPU<float> dS(rc.second, "teste");
            somaColunas(dX.get(), dS.get(), rc.first, rc.second);
            auto S = paraCPU(dS.get(), rc.second);
            for (int c = 0; c < rc.second; ++c) {
                double ref = 0;
                for (int r = 0; r < rc.first; ++r) ref += X[(size_t)r * rc.second + c];
                erro = std::fmax(erro, std::fabs(ref - S[c]));
            }
        }
        VERIFICAR(erro < 1e-4, "soma por coluna: erro %.1e", erro);
    }
    {   // perda MSE (varios blocos)
        const int n = 700;
        auto p = aleatorio(n), a = aleatorio(n);
        auto dp = paraGPU(p), da = paraGPU(a); MemoriaGPU<float> dd(n, "teste"), ds(1, "teste");
        cudaMemset(ds.get(), 0, sizeof(float));
        perdaMSE(dp.get(), da.get(), dd.get(), n, ds.get());
        auto d = paraCPU(dd.get(), n); auto s = paraCPU(ds.get(), 1);
        double sse = 0, erro = 0;
        for (int i = 0; i < n; ++i) {
            double dif = (double)p[i] - a[i];
            sse += dif * dif;
            erro = std::fmax(erro, std::fabs(2 * dif / n - d[i]));
        }
        VERIFICAR(erro < 1e-6 && std::fabs(sse - s[0]) < 1e-3, "perda MSE: gradiente erro %.1e, soma %.4f vs %.4f", erro, s[0], sse);
    }
    {   // montagem do lote
        auto origem = aleatorio(50 * 3);
        std::vector<int> idx(20);
        for (int i = 0; i < 20; ++i) idx[i] = (i * 7) % 50;
        auto dorig = paraGPU(origem); MemoriaGPU<int> didx(20, "teste"); MemoriaGPU<float> ddest(60, "teste");
        cudaMemcpy(didx.get(), idx.data(), 20 * sizeof(int), cudaMemcpyHostToDevice);
        reunirLinhas(dorig.get(), didx.get(), ddest.get(), 20, 3);
        auto dest = paraCPU(ddest.get(), 60);
        bool igual = true;
        for (int r = 0; r < 20; ++r) for (int c = 0; c < 3; ++c) igual &= dest[r * 3 + c] == origem[idx[r] * 3 + c];
        VERIFICAR(igual, "montagem do lote");
    }
    for (Otimizador tipo : {Otimizador::SGD, Otimizador::Adam}) {
        const int n = 1000;
        auto p0 = aleatorio(n), g = aleatorio(n);
        auto dp = paraGPU(p0), dg = paraGPU(g); MemoriaGPU<float> dm(n, "t"), dv(n, "t");
        cudaMemset(dm.get(), 0, n * sizeof(float)); cudaMemset(dv.get(), 0, n * sizeof(float));
        for (int s = 1; s <= 3; ++s) passoOtimizador(tipo, dp.get(), dm.get(), dv.get(), dg.get(), n, 0.01f, s);
        auto p = paraCPU(dp.get(), n);
        double erro = 0;
        for (int i = 0; i < n; ++i) {
            double x = p0[i], m = 0, v = 0;
            for (int s = 1; s <= 3; ++s) {
                if (tipo == Otimizador::SGD) { m = 0.9 * m + g[i]; x -= 0.01 * m; }
                else {
                    m = 0.9 * m + 0.1 * g[i]; v = 0.999 * v + 0.001 * g[i] * g[i];
                    x -= 0.01 * (m / (1 - std::pow(0.9, s))) / (std::sqrt(v / (1 - std::pow(0.999, s))) + 1e-8);
                }
            }
            erro = std::fmax(erro, std::fabs(x - p[i]));
        }
        VERIFICAR(erro < 1e-5, "%s: erro %.1e", tipo == Otimizador::SGD ? "SGD com momento" : "Adam", erro);
    }
}

// ---------------- referencia em double ----------------
struct Referencia {
    std::vector<int> camadas; Ativacao at; Deslocamentos d; std::vector<double> p;
    Referencia(const ModeloMLP& m) : camadas(m.camadas), at(m.ativacao), d(calcularDeslocamentos(m.camadas)),
                                     p(m.parametros.begin(), m.parametros.end()) {}
    std::vector<std::vector<double>> propagar(const std::vector<double>& X, size_t linhas) const {
        const size_t L = camadas.size() - 1;
        std::vector<std::vector<double>> A(L + 1);
        A[0] = X;
        for (size_t l = 0; l < L; ++l) {
            const size_t ni = camadas[l], no = camadas[l + 1];
            const Ativacao a = (l + 1 == L) ? Ativacao::Identidade : at;
            A[l + 1].assign(linhas * no, 0);
            for (size_t r = 0; r < linhas; ++r)
                for (size_t j = 0; j < no; ++j) {
                    double z = p[d.b[l] + j];
                    for (size_t k = 0; k < ni; ++k) z += A[l][r * ni + k] * p[d.W[l] + k * no + j];
                    A[l + 1][r * no + j] = f(z, a);
                }
        }
        return A;
    }
    double perda(const std::vector<double>& X, const std::vector<double>& Y, size_t linhas) const {
        auto A = propagar(X, linhas);
        double s = 0;
        for (size_t i = 0; i < Y.size(); ++i) { double e = A.back()[i] - Y[i]; s += e * e; }
        return s / Y.size();
    }
    std::vector<double> gradiente(const std::vector<double>& X, const std::vector<double>& Y, size_t linhas) const {
        const size_t L = camadas.size() - 1;
        auto A = propagar(X, linhas);
        std::vector<double> g(p.size(), 0), delta(Y.size());
        for (size_t i = 0; i < Y.size(); ++i) delta[i] = 2 * (A[L][i] - Y[i]) / Y.size();
        for (size_t l = L; l-- > 0;) {
            const size_t ni = camadas[l], no = camadas[l + 1];
            for (size_t r = 0; r < linhas; ++r)
                for (size_t j = 0; j < no; ++j) {
                    g[d.b[l] + j] += delta[r * no + j];
                    for (size_t k = 0; k < ni; ++k) g[d.W[l] + k * no + j] += A[l][r * ni + k] * delta[r * no + j];
                }
            if (l > 0) {
                std::vector<double> ant(linhas * ni, 0);
                for (size_t r = 0; r < linhas; ++r)
                    for (size_t k = 0; k < ni; ++k) {
                        double s = 0;
                        for (size_t j = 0; j < no; ++j) s += delta[r * no + j] * p[d.W[l] + k * no + j];
                        ant[r * ni + k] = s * df(A[l][r * ni + k], at);
                    }
                delta.swap(ant);
            }
        }
        return g;
    }
};

static void dadosSin2(int n, std::vector<float>& x, std::vector<float>& y) {
    x.resize(n); y.resize(n);
    for (int i = 0; i < n; ++i) { x[i] = (float)i / (n - 1); y[i] = std::sin(x[i]) * std::sin(x[i]); }
}

static void testeReferenciaDiferencasFinitas() {
    std::printf("referencia vs diferencas finitas\n");
    for (Ativacao at : {Ativacao::Tanh, Ativacao::Sigmoide}) {
        ModeloMLP m = criarMLP({2, 5, 4, 2}, at, 7);
        for (auto& v : m.parametros) v += 0.1f * std::normal_distribution<float>(0, 1)(rng);
        Referencia ref(m);
        std::vector<double> X(12), Y(12);
        for (auto& v : X) v = std::uniform_real_distribution<double>(-1, 1)(rng);
        for (auto& v : Y) v = std::uniform_real_distribution<double>(-1, 1)(rng);
        auto g = ref.gradiente(X, Y, 6);
        double erro = 0;
        for (size_t i = 0; i < ref.p.size(); ++i) {
            const double h = 1e-6, guardado = ref.p[i];
            ref.p[i] = guardado + h; double mais = ref.perda(X, Y, 6);
            ref.p[i] = guardado - h; double menos = ref.perda(X, Y, 6);
            ref.p[i] = guardado;
            erro = std::fmax(erro, std::fabs((mais - menos) / (2 * h) - g[i]));
        }
        VERIFICAR(erro < 1e-7, "ativacao %d: max |analitico - numerico| = %.1e", (int)at, erro);
    }
}

static void testeTreinoIgualReferencia(Otimizador tipo, Ativacao at, int lote, int epocas) {
    const std::vector<int> camadas = {1, 12, 7, 1};
    const int amostras = 37;  // nao e multiplo do lote: exercita o ultimo lote incompleto
    std::vector<float> x, y;
    dadosSin2(amostras, x, y);
    ConfiguracaoTreino cfg;
    cfg.epocas = epocas; cfg.tamanhoLote = lote; cfg.otimizador = tipo; cfg.semente = 99;
    cfg.taxaAprendizado = tipo == Otimizador::Adam ? 0.01f : 0.1f;

    ModeloMLP modelo = criarMLP(camadas, at, 42);
    Referencia ref(modelo);
    auto perdas = treinarMLP(modelo, x, y, cfg);

    std::mt19937 gerador(cfg.semente);
    std::vector<int> ordem(amostras);
    std::iota(ordem.begin(), ordem.end(), 0);
    std::vector<double> m(ref.p.size(), 0), v(ref.p.size(), 0), perdasRef;
    int passo = 0;
    for (int e = 0; e < epocas; ++e) {
        std::shuffle(ordem.begin(), ordem.end(), gerador);
        double sse = 0;
        for (int s = 0; s < amostras; s += lote) {
            const int linhas = std::min(lote, amostras - s);
            std::vector<double> xb(linhas), yb(linhas);
            for (int r = 0; r < linhas; ++r) { xb[r] = x[ordem[s + r]]; yb[r] = y[ordem[s + r]]; }
            sse += ref.perda(xb, yb, linhas) * linhas;
            auto g = ref.gradiente(xb, yb, linhas);
            ++passo;
            for (size_t i = 0; i < g.size(); ++i) {
                if (tipo == Otimizador::Adam) {
                    m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
                    ref.p[i] -= cfg.taxaAprendizado * (m[i] / (1 - std::pow(0.9, passo))) /
                                (std::sqrt(v[i] / (1 - std::pow(0.999, passo))) + 1e-8);
                } else {
                    m[i] = 0.9 * m[i] + g[i];
                    ref.p[i] -= cfg.taxaAprendizado * m[i];
                }
            }
        }
        perdasRef.push_back(sse / amostras);
    }
    double erroP = 0, erroPerda = 0;
    for (size_t i = 0; i < ref.p.size(); ++i) erroP = std::fmax(erroP, std::fabs(modelo.parametros[i] - ref.p[i]));
    for (int e = 0; e < epocas; ++e) erroPerda = std::fmax(erroPerda, std::fabs(perdas[e] - perdasRef[e]) / perdasRef[e]);
    VERIFICAR(erroP < 2e-4 && erroPerda < 1e-4,
              "%s, ativacao %d, lote %d, %d passos: max |dp| = %.1e, perda por epoca erro rel. %.1e",
              tipo == Otimizador::Adam ? "Adam" : "SGD", (int)at, lote, passo, erroP, erroPerda);
}

static void testePrever() {
    std::printf("preverMLP\n");
    ModeloMLP m = criarMLP({3, 20, 2}, Ativacao::Tanh, 5);
    auto X = aleatorio(3 * 1500);  // mais de 1024 amostras: exercita os pedacos
    auto P = preverMLP(m, X);
    Referencia ref(m);
    auto R = ref.propagar(std::vector<double>(X.begin(), X.end()), 1500).back();
    double erro = 0;
    for (size_t i = 0; i < R.size(); ++i) erro = std::fmax(erro, std::fabs(R[i] - P[i]));
    VERIFICAR(P.size() == 3000 && erro < 1e-5, "1500 amostras, 2 saidas: erro %.1e", erro);
}

static void testeValidacoes() {
    std::printf("validacoes de entrada\n");
    auto lanca = [](auto fn) { try { fn(); return false; } catch (const std::runtime_error&) { return true; } };
    ModeloMLP m = criarMLP({2, 4, 1}, Ativacao::Tanh, 1);
    std::vector<float> x(10, 0.f), y(5, 0.f);
    ConfiguracaoTreino cfg;
    VERIFICAR(lanca([&] { preverMLP(m, std::vector<float>(3)); }), "entradas que nao formam amostras inteiras");
    VERIFICAR(lanca([&] { auto mm = m; treinarMLP(mm, x, std::vector<float>(4), cfg); }), "entradas e alvos com quantidades diferentes");
    VERIFICAR(lanca([&] { auto mm = m; mm.parametros.pop_back(); preverMLP(mm, x); }), "parametros incompativeis com as camadas");
    VERIFICAR(lanca([&] { auto c = cfg; c.tamanhoLote = 0; auto mm = m; treinarMLP(mm, x, y, c); }), "lote invalido");
    VERIFICAR(lanca([&] { criarMLP({3}, Ativacao::Tanh, 1); }), "rede com uma camada so");
}

static void testeConvergencia() {
    std::printf("convergencia em sin^2 (variancia do alvo = 0.0505)\n");
    std::vector<float> x, y;
    dadosSin2(100, x, y);
    struct Caso { std::vector<int> camadas; Ativacao at; Otimizador opt; float taxa; unsigned semente; const char* nome; };
    const Caso casos[] = {
        {{1, 8, 8, 8, 1}, Ativacao::Tanh, Otimizador::Adam, 0.01f, 0, "[1,8,8,8,1] tanh Adam, semente 0"},
        {{1, 8, 8, 8, 1}, Ativacao::Tanh, Otimizador::Adam, 0.01f, 1, "[1,8,8,8,1] tanh Adam, semente 1"},
        {{1, 8, 8, 8, 1}, Ativacao::Tanh, Otimizador::Adam, 0.01f, 2, "[1,8,8,8,1] tanh Adam, semente 2"},
        {{1, 32, 1}, Ativacao::Sigmoide, Otimizador::Adam, 0.01f, 0, "[1,32,1] sigmoide Adam"},
        {{1, 16, 16, 1}, Ativacao::ReLU, Otimizador::Adam, 0.01f, 0, "[1,16,16,1] ReLU Adam"},
        {{1, 16, 16, 1}, Ativacao::Tanh, Otimizador::SGD, 0.05f, 0, "[1,16,16,1] tanh SGD com momento"},
    };
    for (const Caso& c : casos) {
        ModeloMLP m = criarMLP(c.camadas, c.at, c.semente);
        ConfiguracaoTreino cfg;
        cfg.otimizador = c.opt; cfg.taxaAprendizado = c.taxa; cfg.semente = c.semente;
        treinarMLP(m, x, y, cfg);
        auto p = preverMLP(m, x);
        double mse = 0;
        for (int i = 0; i < 100; ++i) mse += (p[i] - y[i]) * (p[i] - y[i]);
        mse /= 100;
        VERIFICAR(mse < 1e-3, "%s: MSE final %.1e", c.nome, mse);
    }
}

int main() {
    testeGemm();
    testeDemaisKernels();
    testeReferenciaDiferencasFinitas();
    std::printf("treinarMLP vs referencia\n");
    testeTreinoIgualReferencia(Otimizador::SGD, Ativacao::Tanh, 37, 5);
    testeTreinoIgualReferencia(Otimizador::SGD, Ativacao::Sigmoide, 8, 5);
    testeTreinoIgualReferencia(Otimizador::Adam, Ativacao::Tanh, 8, 5);
    testeTreinoIgualReferencia(Otimizador::Adam, Ativacao::ReLU, 5, 3);
    testePrever();
    testeValidacoes();
    testeConvergencia();
    std::printf(falhas ? "TESTES FALHARAM (%d)\n" : "TODOS OS TESTES PASSARAM\n", falhas);
    return falhas ? 1 : 0;
}
