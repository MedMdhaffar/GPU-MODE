NVCC = nvcc

NVCC_FLAGS = -O0 -arch=sm_86 --compiler-options "/EHsc"
LD_FLAGS = -lcudart

EXE = vector_addition.exe
OBJ = vector_addition.o

all: $(EXE)

$(OBJ): vector_addition.cu
	$(NVCC) -c $< -o $@ $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $^ -o $@ $(LD_FLAGS)

clean:
	del $(OBJ) $(EXE)