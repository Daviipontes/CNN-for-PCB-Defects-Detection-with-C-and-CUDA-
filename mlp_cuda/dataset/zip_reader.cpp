#include "dataset/zip_reader.h"

#include <zlib.h>

#include <algorithm>
#include <stdexcept>

namespace {

std::uint16_t u16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }

std::uint32_t u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t u64(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(u32(p)) | (static_cast<std::uint64_t>(u32(p + 4)) << 32);
}

constexpr std::uint64_t LIMITE_ENTRADA = 1u << 30;  // recusa entradas com mais de 1 GB

}  // namespace

LeitorZip::LeitorZip(const std::string& caminho) : arquivo(caminho, std::ios::binary) {
    if (!arquivo) throw std::runtime_error("Nao foi possivel abrir " + caminho);
    lerDiretorioCentral();
}

std::vector<std::string> LeitorZip::nomes() const {
    std::vector<std::string> n;
    for (const auto& par : entradas) n.push_back(par.first);
    return n;
}

std::vector<std::uint8_t> LeitorZip::ler(const std::string& nome) {
    const auto it = entradas.find(nome);
    if (it == entradas.end()) throw std::runtime_error("Nao encontrado no zip: " + nome);
    const Entrada& e = it->second;

    // Cabecalho local: o tamanho do nome e do campo extra pode diferir do diretorio central.
    std::uint8_t local[30];
    lerBytes(e.deslocamento, local, sizeof(local));
    if (u32(local) != 0x04034b50) throw std::runtime_error("Cabecalho local invalido: " + nome);
    const std::uint64_t dados = e.deslocamento + 30 + u16(local + 26) + u16(local + 28);

    if (e.tamComprimido > LIMITE_ENTRADA || e.tamOriginal > LIMITE_ENTRADA)
        throw std::runtime_error("Entrada grande demais: " + nome);
    std::vector<std::uint8_t> comprimido(e.tamComprimido);
    if (!comprimido.empty()) lerBytes(dados, comprimido.data(), comprimido.size());
    if (e.metodo == 0) return comprimido;
    if (e.metodo != 8) throw std::runtime_error("Metodo de compressao nao suportado: " + nome);

    std::vector<std::uint8_t> saida(e.tamOriginal);
    z_stream fluxo{};
    if (inflateInit2(&fluxo, -MAX_WBITS) != Z_OK) throw std::runtime_error("inflateInit falhou");
    fluxo.next_in = comprimido.data();
    fluxo.avail_in = static_cast<uInt>(comprimido.size());
    fluxo.next_out = saida.data();
    fluxo.avail_out = static_cast<uInt>(saida.size());
    const int r = inflate(&fluxo, Z_FINISH);
    inflateEnd(&fluxo);
    if (r != Z_STREAM_END) throw std::runtime_error("Falha ao descomprimir: " + nome);
    return saida;
}

void LeitorZip::lerBytes(std::uint64_t posicao, void* destino, size_t n) {
    arquivo.clear();
    arquivo.seekg(static_cast<std::streamoff>(posicao));
    arquivo.read(static_cast<char*>(destino), static_cast<std::streamsize>(n));
    if (!arquivo) throw std::runtime_error("Zip truncado ou corrompido");
}

void LeitorZip::lerDiretorioCentral() {
    arquivo.seekg(0, std::ios::end);
    const std::uint64_t tamanho = static_cast<std::uint64_t>(arquivo.tellg());

    // O registro de fim do diretorio central fica nos ultimos (22 + comentario) bytes.
    const std::uint64_t janela = std::min<std::uint64_t>(tamanho, 22 + 65535);
    std::vector<std::uint8_t> cauda(janela);
    lerBytes(tamanho - janela, cauda.data(), cauda.size());
    for (std::int64_t i = static_cast<std::int64_t>(janela) - 22; i >= 0; --i) {
        if (u32(&cauda[i]) != 0x06054b50) continue;
        const std::uint16_t n = u16(&cauda[i + 10]);
        const std::uint32_t tamDir = u32(&cauda[i + 12]);
        const std::uint32_t posDir = u32(&cauda[i + 16]);
        if (n == 0xFFFF || tamDir == 0xFFFFFFFFu || posDir == 0xFFFFFFFFu)
            throw std::runtime_error("Zip64 no registro final nao suportado");

        std::vector<std::uint8_t> dir(tamDir);
        lerBytes(posDir, dir.data(), dir.size());
        size_t p = 0;
        for (int k = 0; k < n; ++k) {
            if (p + 46 > dir.size() || u32(&dir[p]) != 0x02014b50)
                throw std::runtime_error("Diretorio central do zip invalido");
            const std::uint16_t nomeLen = u16(&dir[p + 28]);
            const std::uint16_t extraLen = u16(&dir[p + 30]);
            const std::uint16_t comentLen = u16(&dir[p + 32]);
            if (p + 46 + nomeLen + extraLen > dir.size())
                throw std::runtime_error("Diretorio central do zip invalido");
            const std::string nome(reinterpret_cast<const char*>(&dir[p + 46]), nomeLen);
            Entrada e{u16(&dir[p + 10]), u32(&dir[p + 20]), u32(&dir[p + 24]), u32(&dir[p + 42])};

            // Campos que valem 0xFFFFFFFF ficam no campo extra zip64 (id 0x0001), nesta ordem:
            // tamanho original, tamanho comprimido, deslocamento.
            if (e.tamOriginal == 0xFFFFFFFFu || e.tamComprimido == 0xFFFFFFFFu || e.deslocamento == 0xFFFFFFFFu) {
                const std::uint8_t* x = &dir[p + 46 + nomeLen];
                size_t q = 0;
                while (q + 4 <= extraLen) {
                    const std::uint16_t id = u16(x + q), tam = u16(x + q + 2);
                    if (q + 4 + tam > extraLen) break;
                    if (id == 0x0001) {
                        size_t o = q + 4;
                        const size_t fim = q + 4 + tam;
                        for (std::uint64_t* campo : {&e.tamOriginal, &e.tamComprimido, &e.deslocamento}) {
                            if (*campo != 0xFFFFFFFFu) continue;
                            if (o + 8 > fim) throw std::runtime_error("Campo zip64 invalido: " + nome);
                            *campo = u64(x + o);
                            o += 8;
                        }
                        break;
                    }
                    q += 4 + tam;
                }
            }
            if (!nome.empty() && nome.back() != '/') entradas[nome] = e;
            p += 46 + nomeLen + extraLen + comentLen;
        }
        return;
    }
    throw std::runtime_error("Arquivo nao e um zip valido");
}
