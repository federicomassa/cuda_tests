.PHONY: directories

all: directories add device_info

directories: bin
bin:
	mkdir bin

add: add.cu directories
	nvcc -o bin/add add.cu -std=c++11

device_info: device_info.cu directories
	nvcc -o bin/device_info device_info.cu -std=c++11
