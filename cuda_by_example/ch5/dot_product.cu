#include "../common/book.h"
#include <iostream>
#include <chrono>
#include <sstream>

const long N = 400000;
const int THREADS = 256;
const int BLOCKS = std::ceil(double(N)/double(THREADS));

void dot(int* a, int* b, int* sum) {
  int tid = 0;
  *sum = 0;
  
  while (tid < N) {
    *sum += a[tid]*b[tid];
    tid++;
  }
}

__global__
void dot_gpu(int* a, int* b, int* c) {
  
  __shared__ float cache[THREADS];
  
  float tmp_sum = 0;
  
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  while (tid < N) { 
    tmp_sum = a[tid]*b[tid];
    tid += blockDim.x*gridDim.x;
  }
  
  int cacheIndex = threadIdx.x;
  cache[cacheIndex] = tmp_sum;
  
  // wait for all the threads
  __syncthreads();
  
  // Now use these threads to sum the elements in parallel
  int i = THREADS/2;
  
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    
    i = i/2;
    __syncthreads();
  }

  if (cacheIndex == 0)
    c[blockIdx.x] = cache[0];
}

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

void cpu_test() {
  int* a = new int[N];
  int* b = new int[N];
  int* sum = new int;
  
  for (int i = 0; i < N; i++) {
    a[i] = -i;
    b[i] = i*i;
  }
  
  printf("Starting CPU benchmark...\n");
  
  auto t0 = Time::now();
  dot(a,b,sum);
  auto t1 = Time::now();
  
  fsec fs = t1 - t0;
  std::cout << "CPU took: " << fs.count() << " s\n";	
  
  delete[] a;
  delete[] b;
  delete sum;
}

void gpu_test() {
  printf("Starting GPU benchmark...\n");
  
  int* a = new int[N];
  int* b = new int[N];
  int* c = new int[BLOCKS];
  
  for (int i = 0; i < N; i++) {
    a[i] = -i;
    b[i] = i*i;
  }

  auto t_init = Time::now();
  int* dev_a, *dev_b, *dev_c;
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, BLOCKS*sizeof(int)));
  
  HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_c, c, BLOCKS*sizeof(int), cudaMemcpyHostToDevice));
  

  // COMPUTE PARTIAL SUM
  auto t0 = Time::now();
  
  dot_gpu<<<BLOCKS,THREADS>>>(dev_a, dev_b, dev_c);

  // TRANSFER
  auto t1 = Time::now();
  
  HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
  
  auto t2 = Time::now();
  // FINAL SUM
  int sum = 0;
  for (int i = 0; i < BLOCKS; i++)
    sum += c[i];

  auto t3 = Time::now();

  fsec fs_init = t0 - t_init;
  fsec fs = t1 - t0;
  fsec fs2 = t2 - t1;
  fsec fs3 = t3 - t2;
  
  printf("GPU took %f s (%f to load input onto GPU, %f to compute + %f to retrieve data from device + %f to finalize)", (fs + fs2 + fs3).count(), fs_init.count(), fs.count(), fs2.count(), fs3.count());
  
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  
  delete[] a;
  delete[] b;
  delete[] c;
}

int main(int argc, char** argv) {
  std::cout << "----------- Dot product of vectors of size " << N << "-----------" << std::endl;
  
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
