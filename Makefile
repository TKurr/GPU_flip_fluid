CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -std=c++17 -O3 -ffast-math -Wall -Wextra
NVCCFLAGS ?= -std=c++17 -O3 -arch=sm_75 --use_fast_math
LDFLAGS  ?=
LIBS     := -lGL -lGLX -lX11 -lm

# CPU source
SRC := flip_fluid.cpp ui.cpp main.cpp
OBJ := $(SRC:.cpp=.o)
BIN := flip

# CUDA source
CU_SRC := cuda/main_cuda.cu cuda/flip_fluid_cuda.cu ui.cpp
CU_OBJ := cuda/main_cuda.o cuda/flip_fluid_cuda.o ui_cuda.o
BIN_CU := flip_cuda

# Numerical validation (CPU vs CUDA, headless — no GL/X11)
VAL_OBJ := cuda/validate.o cuda/flip_fluid_cuda.o flip_fluid.o
BIN_VAL := flip_validate

all: $(BIN) $(BIN_CU)

# CPU build
$(BIN): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp flip_fluid.h ui.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# CUDA build
$(BIN_CU): $(CU_OBJ)
	$(NVCC) $(LDFLAGS) -o $@ $^ $(LIBS)

cuda/main_cuda.o: cuda/main_cuda.cu cuda/flip_fluid_cuda.cuh ui.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

cuda/flip_fluid_cuda.o: cuda/flip_fluid_cuda.cu cuda/flip_fluid_cuda.cuh cuda/kernels.cuh
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

ui_cuda.o: ui.cpp ui.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Validation build (links the g++-compiled CPU object with CUDA objects)
$(BIN_VAL): $(VAL_OBJ)
	$(NVCC) $(LDFLAGS) -o $@ $^ -lm

cuda/validate.o: cuda/validate.cu cuda/flip_fluid_cuda.cuh cuda/kernels.cuh flip_fluid.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(BIN) $(CU_OBJ) $(BIN_CU) cuda/validate.o $(BIN_VAL)

.PHONY: all clean
