.PHONY: directories

all: directories dot_product

directories: bin
bin:
	mkdir bin

dot_product: dot_product.cu directories
	nvcc -o bin/dot_product dot_product.cu -std=c++11
