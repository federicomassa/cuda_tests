.PHONY: directories

all: directories sum_vec more_sums

directories: bin
bin:
	mkdir bin

sum_vec: sum_vec.cu directories
	nvcc -o bin/sum_vec sum_vec.cu -std=c++11

more_sums: more_sums.cu directories
	nvcc -o bin/more_sums more_sums.cu -std=c++11
