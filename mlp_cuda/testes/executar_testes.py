#!/usr/bin/env python3
"""
Compila e executa os testes emulados na CPU (nao precisa de GPU).

Os arquivos .cu (processamento.cu e gpu/*.cu) sao compilados como C++ comum usando testes/emu/cuda_runtime.h. A unica
transformacao no codigo e a dos lancamentos de kernel, que nao sao C++ valido:
    kernel<<<grid, bloco>>>(args);  ->  emu::launch(dim3(grid), dim3(bloco), [&]{ kernel(args); });

Uso (na pasta do projeto):  python3 testes/executar_testes.py      ou      make testes
"""
import os
import random
import re
import struct
import subprocess
import sys
import zipfile
import zlib

TESTES = os.path.dirname(os.path.abspath(__file__))
PROJETO = os.path.dirname(TESTES)
BUILD = os.path.join(TESTES, "build")

LANCAMENTO = re.compile(r"([A-Za-z_][\w:]*(?:<[^<>;]*>)?)<<<(.*?)>>>\((.*)\);")


def separar_argumentos(s):
    partes, nivel, atual = [], 0, ""
    for ch in s:
        if ch in "([<":
            nivel += 1
        elif ch in ")]>":
            nivel -= 1
        if ch == "," and nivel == 0:
            partes.append(atual.strip())
            atual = ""
        else:
            atual += ch
    partes.append(atual.strip())
    return partes


def transformar(codigo):
    saida, n = [], 0
    for linha in codigo.splitlines():
        m = LANCAMENTO.search(linha)
        if m:
            nome, cfg, args = m.groups()
            cfg = separar_argumentos(cfg)
            novo = f"emu::launch(dim3({cfg[0]}), dim3({cfg[1]}), [&]{{ {nome}({args}); }});"
            linha = linha[: m.start()] + novo + linha[m.end():]
            n += 1
        assert "<<<" not in linha, f"lancamento nao tratado: {linha}"
        saida.append(linha)
    return "\n".join(saida) + "\n", n


def compilar(fontes, executavel, flags):
    subprocess.run(["g++", *flags, *fontes, "-o", executavel], check=True)


def png_cinza(largura, altura, pixels):
    """PNG 8 bits em tons de cinza (so zlib e struct)."""
    def bloco(tipo, dados):
        corpo = tipo + dados
        return struct.pack(">I", len(dados)) + corpo + struct.pack(">I", zlib.crc32(corpo))
    linhas = b"".join(b"\x00" + bytes(pixels[y * largura:(y + 1) * largura]) for y in range(altura))
    return (b"\x89PNG\r\n\x1a\n" + bloco(b"IHDR", struct.pack(">IIBBBBB", largura, altura, 8, 0, 0, 0, 0))
            + bloco(b"IDAT", zlib.compress(linhas)) + bloco(b"IEND", b""))


def gerar_zip_sintetico(caminho, classes=6, imagens_por_classe=12, lado=128):
    """Cria um zip no mesmo formato do PCB_DATASET: Annotations/<classe>/*.xml (caixas) e
    images/<classe>/*.png. Cada classe tem uma textura diferente dentro das caixas."""
    rng = random.Random(0)
    with zipfile.ZipFile(caminho, "w", zipfile.ZIP_DEFLATED) as z:
        for c in range(classes):
            for k in range(imagens_por_classe):
                pixels = [rng.randint(60, 90) for _ in range(lado * lado)]
                objetos = ""
                for _ in range(2):
                    x0, y0 = rng.randint(10, lado - 50), rng.randint(10, lado - 50)
                    for y in range(24):
                        for x in range(24):
                            if (x * (c + 1) + y) % 7 < 3:
                                pixels[(y0 + y) * lado + x0 + x] = 220 + rng.randint(-20, 20)
                    objetos += (f"<object><name>d</name><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin>"
                                f"<xmax>{x0 + 24}</xmax><ymax>{y0 + 24}</ymax></bndbox></object>")
                nome = f"img_{k:02d}"
                xml = (f"<annotation><folder>classe_{c}</folder><filename>{nome}.png</filename>"
                       f"{objetos}</annotation>")
                z.writestr(f"PCB_DATASET/Annotations/classe_{c}/{nome}.xml", xml)
                z.writestr(f"PCB_DATASET/images/classe_{c}/{nome}.png", png_cinza(lado, lado, pixels))


def fontes_cu():
    """Todos os .cu do projeto: processamento.cu e gpu/*.cu."""
    achados = [os.path.join(PROJETO, "processamento.cu")]
    pasta = os.path.join(PROJETO, "gpu")
    achados += sorted(os.path.join(pasta, f) for f in os.listdir(pasta) if f.endswith(".cu"))
    return achados


def fontes_cpp():
    """Codigo de CPU do projeto: main.cpp, avaliacao.cpp e dataset/*.cpp."""
    pasta = os.path.join(PROJETO, "dataset")
    return ([os.path.join(PROJETO, "main.cpp"), os.path.join(PROJETO, "avaliacao.cpp")]
            + sorted(os.path.join(pasta, f) for f in os.listdir(pasta) if f.endswith(".cpp")))


def emular_cu():
    """Converte cada .cu em testes/build/<nome>_emu.cpp e devolve os caminhos gerados."""
    gerados, total = [], 0
    for fonte in fontes_cu():
        with open(fonte) as f:
            codigo, n = transformar(f.read())
        nome = os.path.relpath(fonte, PROJETO).replace(os.sep, "_")[:-3] + "_emu.cpp"
        caminho = os.path.join(BUILD, nome)
        with open(caminho, "w") as f:
            f.write(f'#line 1 "{fonte}"\n' + codigo)
        gerados.append(caminho)
        total += n
    print(f"[build] {len(gerados)} arquivos .cu emulados, {total} lancamentos de kernel convertidos")
    return gerados


def main():
    os.makedirs(BUILD, exist_ok=True)
    emulados = emular_cu()

    flags = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Wno-unknown-pragmas",
             "-I", os.path.join(TESTES, "emu"), "-I", PROJETO, "-I", BUILD]
    if os.environ.get("SANITIZE"):
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]

    # 1) Testes de kernels e da rede (ligados aos modulos da GPU emulados).
    exe = os.path.join(BUILD, "teste_mlp")
    compilar([os.path.join(TESTES, "teste_mlp.cpp"), *emulados], exe, flags)
    ok = subprocess.run([exe]).returncode == 0

    # 2) O programa principal completo, emulado, executado dentro de testes/build.
    exe_main = os.path.join(BUILD, "mlp_cuda_emu")
    compilar([*emulados, *fontes_cpp(), "-lz"], exe_main, flags)
    print("[run] main.cpp emulado (zip sintetico pequeno, 30 epocas)")
    gerar_zip_sintetico(os.path.join(BUILD, "dataset_teste.zip"))
    ok &= subprocess.run([exe_main, "dataset_teste.zip", "30", "16"], cwd=BUILD).returncode == 0

    print("TUDO OK" if ok else "HOUVE FALHAS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
