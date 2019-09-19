.PHONY: directories

all: directories sum_vec

directories: bin
bin:
	mkdir bin

sum_vec: sum_vec.cu directories
	nvcc -o bin/sum_vec sum_vec.cu -std=c++11
