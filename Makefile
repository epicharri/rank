INCLUDE_PATH = ../include
SOURCE_PATH = src/search.cpp
BUILD_PATH = build/search

NVCC_FLAGS = -Xcompiler "-fopenmp,-lgomp,-march=native, -O3" -gencode arch=compute_80,code=sm_80 -O3 --include-path $(INCLUDE_PATH) -x cu -g

NVCC = nvcc

.ONESHELL:

all: $(BUILD_PATH)

$(BUILD_PATH): $(SOURCE_PATH) $(wildcard $(INCLUDE_PATH)/*.hpp)
	mkdir -p $(dir $@)
	module load gcc
	module load cuda
	if $(NVCC) $(NVCC_FLAGS) -o $@ $(SOURCE_PATH); then \
		cuobjdump -ptx $@ > $@.ptx; \
		cuobjdump -sass $@ > $@.sass; \
	fi

clean:
	rm -rf $(BUILD_PATH) $(BUILD_PATH).ptx $(BUILD_PATH).sass
