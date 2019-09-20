.PHONY: directories

all: directories pseudo_particle

directories: bin
bin:
	mkdir bin

pseudo_particle: pseudo_particle.cu directories
	nvcc -o bin/pseudo_particle pseudo_particle.cu -std=c++11
