CXX := g++
NVCC := nvcc
# Opcional: arquitetura da GPU, ex.: make ARCH=sm_86
NVCC_ARCH := $(if $(ARCH),-arch=$(ARCH),)

# Codigo da CPU (g++): programa principal, avaliacao e leitura do dataset.
SRC_CPP := main.cpp avaliacao.cpp $(wildcard dataset/*.cpp)
# Codigo da GPU (nvcc): interface publica e modulos de gpu/.
SRC_CU := processamento.cu $(wildcard gpu/*.cu)

OBJ := $(patsubst %.cpp,obj/%.o,$(SRC_CPP)) $(patsubst %.cu,obj/%.o,$(SRC_CU))
# Qualquer cabecalho alterado recompila tudo: simples e suficiente para um projeto pequeno.
HEADERS := $(wildcard *.h dataset/*.h gpu/*.h gpu/*.cuh third_party/*.h)

all: mlp_cuda

mlp_cuda: $(OBJ)
	$(NVCC) $(OBJ) -lz -o mlp_cuda

obj/%.o: %.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	$(CXX) -O3 -std=c++17 -I. -c $< -o $@

obj/%.o: %.cu $(HEADERS)
	@mkdir -p $(dir $@)
	$(NVCC) -O3 -std=c++17 -I. $(NVCC_ARCH) -c $< -o $@

testes:
	python3 testes/executar_testes.py

clean:
	rm -rf obj mlp_cuda saida.csv testes/build

.PHONY: all clean testes
