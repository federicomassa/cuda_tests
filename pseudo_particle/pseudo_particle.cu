// ============================================ //
// Author: Federico Massa
//
// This is a CPU/GPU particle filter benchmark
// It doesn't actually do anything but replicate
// computations similar to a real particle filter,
// just to evaluate CPU/GPU performances.
// ============================================ //

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <numeric>
#include <random>
#include <chrono>

int nParticles;
int sizeX, sizeY;
int laserPoints;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

std::vector<int> closest_idx;

void generateLaserPoints(std::vector<float>& laserX, std::vector<float>& laserY)
{
  for (int i = 0; i < laserX.size(); i++) {
    laserX[i] = rand()/RAND_MAX;
  }
  for (int i = 0; i < laserY.size(); i++) {
    laserY[i] = rand()/RAND_MAX;
  }
}

void generateOdometry(float& vx, float& vy) {
  vx = rand()/RAND_MAX;
  vy = rand()/RAND_MAX;
}

__host__ __device__ void getPixel(const float& laserX, const float& laserY, int& laserPX, int& laserPY) {
  laserPX = 0;
  laserPY = 0;
}

__host__ __device__ float computeWeight(const int& laserPX, const int& laserPY, const int& sizeX, int* closest_idx) {
  int idx = laserPX + sizeX*laserPY;

  int w = closest_idx[idx];
  int wx = w*0;
  int wy = w*0;
  
  return wx*wx + wy*wy;
}

__global__ void pf_iteration_dev(int* dev_closest_idx, float* dev_laserX, float* dev_laserY, float* dev_vx, float* dev_vy, float* dev_particles_x, float* dev_particles_y, float* dev_particles_theta, int* dev_laserPoints, float* dev_weights, int* dev_sizeX, int* dev_sizeY) {
  int index = threadIdx.x + blockIdx.x*blockDim.x;

  // Predict
  dev_particles_x[index] += *dev_vx*cos(dev_particles_theta[index]) + *dev_vy*sin(dev_particles_theta[index]);
  dev_particles_x[index] += *dev_vx*cos(dev_particles_theta[index]) + *dev_vy*sin(dev_particles_theta[index]);
  dev_particles_theta[index] += 1*3.14159/180.0;

  // Update
  float weight = 0;
  for (int i = 0; i < *dev_laserPoints; i++) {
    float localLaserX = dev_laserX[i]*cos(dev_particles_theta[index]) + dev_laserY[i]*sin(dev_particles_theta[index]);
    float localLaserY = -dev_laserX[i]*sin(dev_particles_theta[index]) + dev_laserY[i]*cos(dev_particles_theta[index]);

    int localLaserPX, localLaserPY;

    // Transform laser point to pixel coordinates
    getPixel(localLaserX, localLaserY, localLaserPX, localLaserPY);

    weight += computeWeight(localLaserPX, localLaserPY, *dev_sizeX, dev_closest_idx);
  }

  dev_weights[index] = weight;
}

void pf_iteration(std::vector<int>& closest_idx, const std::vector<float>& currentlaserX, const std::vector<float>& currentlaserY, const float& vx, const float& vy, std::vector<float>& init_particles_x, std::vector<float>& init_particles_y, std::vector<float>& init_particles_theta, std::vector<float>& weights) {
  for (int index = 0; index < nParticles; index++) {
  
    // Predict
    init_particles_x[index] += vx*cos(init_particles_theta[index]) + vy*sin(init_particles_theta[index]);
    init_particles_y[index] += -vx*sin(init_particles_theta[index]) + vy*cos(init_particles_theta[index]);
    init_particles_theta[index] += 1*3.14159/180.0;
    
    // Update
    float weight = 0;
    for (int i = 0; i < laserPoints; i++) {
      float localLaserX = currentlaserX[i]*cos(init_particles_theta[index]) + currentlaserY[i]*sin(init_particles_theta[index]);
      float localLaserY = -currentlaserX[i]*sin(init_particles_theta[index]) + currentlaserY[i]*cos(init_particles_theta[index]);
      
      int localLaserPX, localLaserPY;
      
      // Transform laser point to pixel coordinates
      getPixel(localLaserX, localLaserY, localLaserPX, localLaserPY);
      
      weight += computeWeight(localLaserPX, localLaserPY, sizeX, closest_idx.data());
    }

    weights[index] = weight;
  }
}

void gpu_test() {
  int* dev_closest_idx;
  int* dev_laserPoints;
  float *dev_laserX, *dev_laserY;
  float *dev_vx, *dev_vy;

  float* dev_particles_x;
  float* dev_particles_y;
  float* dev_particles_theta;
  float* dev_weights;

  int* dev_sizeX, *dev_sizeY;
  
  // Allocate distance map and copy it from host
  cudaMalloc((void**)&dev_closest_idx, sizeX*sizeY*sizeof(int));
  cudaMemcpy(dev_closest_idx, closest_idx.data(), sizeX*sizeY*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dev_laserX, laserPoints*sizeof(float));
  cudaMalloc((void**)&dev_laserY, laserPoints*sizeof(float));
  cudaMalloc((void**)&dev_vx, sizeof(float));
  cudaMalloc((void**)&dev_vy, sizeof(float));
  cudaMalloc((void**)&dev_particles_x, nParticles*sizeof(float));
  cudaMalloc((void**)&dev_particles_y, nParticles*sizeof(float));
  cudaMalloc((void**)&dev_particles_theta, nParticles*sizeof(float));
  cudaMalloc((void**)&dev_weights, nParticles*sizeof(float));
  cudaMalloc((void**)&dev_laserPoints, sizeof(int));
  cudaMalloc((void**)&dev_sizeX, sizeof(int));
  cudaMalloc((void**)&dev_sizeY, sizeof(int));
  cudaMemcpy(dev_sizeX, &sizeX, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_sizeY, &sizeY, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_laserPoints, &laserPoints, sizeof(int), cudaMemcpyHostToDevice);

  
  std::vector<float> init_particles_x, init_particles_y, init_particles_theta;
  init_particles_x.reserve(nParticles);
  init_particles_y.reserve(nParticles);
  init_particles_theta.reserve(nParticles);

  // Initialize particles
  for (int i = 0; i < nParticles; i++) {
    init_particles_x[i] = init_particles_y[i] = init_particles_theta[i] = 0.0;
  }

  cudaMemcpy(dev_particles_x, init_particles_x.data(), nParticles*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_particles_y, init_particles_y.data(), nParticles*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_particles_theta, init_particles_theta.data(), nParticles*sizeof(float), cudaMemcpyHostToDevice);
  
  // Current measurements
  std::vector<float> currentLaserX(laserPoints);
  std::vector<float> currentLaserY(laserPoints);
  float vx, vy;

  // Weights (output)
  std::vector<float> weights(nParticles);

  const int iterations = 100;
  std::vector<float> alloc_time, compute_time, retrieve_time;
  // Particle filter iteration
  for (int i = 0; i < iterations; i++) {
    auto t0 = Time::now();
    
    generateLaserPoints(currentLaserX, currentLaserY);
    generateOdometry(vx, vy);

    // Transfer current measurements to device
    cudaMemcpy(dev_laserX, currentLaserX.data(), laserPoints*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_laserY, currentLaserY.data(), laserPoints*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vx, &vx, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vy, &vy, sizeof(float), cudaMemcpyHostToDevice);

    // 1 particle for each core
    const int THREADS = 128;
    const int BLOCKS = std::ceil(float(nParticles)/float(THREADS));

    auto t1 = Time::now();
    
    // Launch an iteration
    
    pf_iteration_dev<<<BLOCKS,THREADS>>>(dev_closest_idx, dev_laserX, dev_laserY, dev_vx, dev_vy, dev_particles_x, dev_particles_y, dev_particles_theta, dev_laserPoints, dev_weights, dev_sizeX, dev_sizeY);
    cudaDeviceSynchronize();

    auto t2 = Time::now();
   
    // Retrieve results
    cudaMemcpy(weights.data(), dev_weights, nParticles*sizeof(float), cudaMemcpyDeviceToHost);

    auto t3 = Time::now();

    fsec fs_alloc = t1 - t0;
    fsec fs_compute = t2 - t1;
    fsec fs_retrieve = t3 - t2;
    
    float alloc = fs_alloc.count();
    float compute = fs_compute.count();
    float retrieve = fs_retrieve.count();

    alloc_time.push_back(alloc);
    compute_time.push_back(compute);
    retrieve_time.push_back(retrieve);
  }

  float alloc_mean = std::accumulate(alloc_time.begin(), alloc_time.end(), 0.0)/float(iterations);
  float compute_mean = std::accumulate(compute_time.begin(), compute_time.end(), 0.0)/float(iterations);
  float retrieve_mean = std::accumulate(retrieve_time.begin(), retrieve_time.end(), 0.0)/float(iterations);
  
  printf("GPU test finished, average time over %d iterations was %f: %f (alloc), %f (compute), %f (retrieve)\n", iterations, alloc_mean+compute_mean+retrieve_mean, alloc_mean, compute_mean, retrieve_mean);
  

  // Release memory on device
  cudaFree(dev_closest_idx);
  cudaFree(dev_laserX);
  cudaFree(dev_laserY);
  cudaFree(dev_vx);
  cudaFree(dev_vy);
}


void cpu_test() {  
  std::vector<float> init_particles_x, init_particles_y, init_particles_theta;
  init_particles_x.reserve(nParticles);
  init_particles_y.reserve(nParticles);
  init_particles_theta.reserve(nParticles);

  // Initialize particles
  for (int i = 0; i < nParticles; i++) {
    init_particles_x[i] = init_particles_y[i] = init_particles_theta[i] = 0.0;
  }
  
  // Current measurements
  std::vector<float> currentLaserX(laserPoints);
  std::vector<float> currentLaserY(laserPoints);
  float vx, vy;

  // Weights (output)
  std::vector<float> weights(nParticles);

  const int iterations = 100;
  std::vector<float> tot_time;
  // Particle filter iteration
  for (int i = 0; i < iterations; i++) {
    auto t0 = Time::now();
    
    generateLaserPoints(currentLaserX, currentLaserY);
    generateOdometry(vx, vy);

    // Launch an iteration
    
    pf_iteration(closest_idx, currentLaserX, currentLaserY, vx, vy, init_particles_x, init_particles_y, init_particles_theta, weights);

  
    auto t3 = Time::now();

    fsec fs_tot = t3 - t0;
    float tot = fs_tot.count();

    tot_time.push_back(tot);
  }

  float tot_mean = std::accumulate(tot_time.begin(), tot_time.end(), 0.0)/float(iterations);

  printf("###########################################################\n");
  printf("CPU test finished, average time over %d iterations was %f\n", iterations, tot_mean);
  printf("###########################################################\n\n");
}


int main(int argc, char** argv) {
  if (argc != 5) {
    std::cout << "Please specify an integer number of particles, map sizeX, map sizeY, laserPoints" << std::endl;
    exit(1);
  }
  
  std::stringstream ss;
  ss << argv[1];
  ss >> nParticles;
  ss.clear();

  ss << argv[2];
  ss >> sizeX;
  ss.clear();

  ss << argv[3];
  ss >> sizeY;
  ss.clear();

  ss << argv[4];
  ss >> laserPoints;
  ss.clear();

  printf("==================================================================\n");
  printf("Dummy particle filter generator with %d particles, and a %dx%d map\n", nParticles, sizeX, sizeY);
  printf("==================================================================\n\n");
  
  // Allocate dummy map, vectorized, indicating the index of the closest black pixel in the map
  // with respect to the current index
  closest_idx.reserve(sizeX*sizeY);
  for (int i = 0; i < closest_idx.size(); i++)
    closest_idx[i] = rand();

  cpu_test();
  
  int count;
  cudaGetDeviceCount(&count);

  for (int i = 0; i < count; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    printf("###################################################\n");
    printf("Starting GPU benchmark on device %d with name: %s\n", i, prop.name);
    cudaSetDevice(i);

    gpu_test();
    printf("##################################################\n\n");
  }
}