#include "../common/book.h"
#include <iostream>
#include <chrono>
#include <sstream>

const long N = 10000000;
const int THREADS = 512;
const int BLOCKS = std::ceil(double(N)/double(THREADS));
//const int BLOCKS = 512;

void add(int* a, int* b, int* c) {
	int tid = 0;
	
	while (tid < N) {
	  c[tid] =  a[tid] + b[tid];
	  tid++;
	}
}

__global__
void add_gpu(int* a, int* b, int* c) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < N) {
	  c[tid] =  a[tid] + b[tid];
	  tid += blockDim.x*gridDim.x;
	}
}

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

void cpu_test() {
	int* a = new int[N];
	int* b = new int[N];
	int* c = new int[N];
	
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i+2;
	}
	
	printf("Starting CPU benchmark...\n");
	
	auto t0 = Time::now();
	add(a,b,c);
	auto t1 = Time::now();
	
	
	fsec fs = t1 - t0;
	std::cout << "CPU took: " << fs.count() << " s\n";
	std::cout << "Test: central index is: " << c[N/2] << std::endl;

	delete a, b, c;
}

void gpu_test() {
	printf("Starting GPU benchmark...\n");
	
	int* a = new int[N];
	int* b = new int[N];
	int* c = new int[N];
	
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i+2;
	}
	
	int* dev_a, *dev_b, *dev_c;
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(int)));
	
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));
	
	// Add and transfer result to CPU
	auto t0 = Time::now();

	add_gpu<<<BLOCKS,THREADS>>>(dev_a, dev_b, dev_c);
	cudaDeviceSynchronize();
	auto t1 = Time::now();
	
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));
	auto t2 = Time::now();
	
	fsec fs = t1 - t0;
	fsec fs2 = t2 - t1;

	printf("GPU took %f s (%f to sum + %f to retrieve data from device)\n", (fs + fs2).count(), fs.count(), fs2.count());
	std::cout << "Test: central index is: " << c[N/2] << std::endl;
	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	
	delete a,b,c;
}

int main(int argc, char** argv) {
	std::cout << "----------- Summing vectors of size " << N << "-----------" << std::endl;

	cpu_test();
	
	printf("##################################\n\n");
		
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	
	for (int i = 0; i < count; i++) {
		cudaDeviceProp prop;
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
		
		printf("Starting GPU benchmark on device %d with name: %s\n", i, prop.name);
		cudaSetDevice(i);
		
		gpu_test();
		printf("\n###############################\n");
	}
	
	return 0;
}
