#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <new>

#include <curand_kernel.h>

#include "configuration.h"
#include "cuda_helper.h"

// Arrays containing all Body objects on device.
__device__ float* dev_Body_pos_x;
__device__ float* dev_Body_pos_y;
__device__ float* dev_Body_vel_x;
__device__ float* dev_Body_vel_y;
__device__ float* dev_Body_mass;
__device__ float* dev_Body_force_x;
__device__ float* dev_Body_force_y;

float* host_Body_pos_x;
float* host_Body_pos_y;
float* host_Body_vel_x;
float* host_Body_vel_y;
float* host_Body_mass;
float* host_Body_force_x;
float* host_Body_force_y;


__device__ void new_Body(int id, float pos_x, float pos_y,
                         float vel_x, float vel_y, float mass) {
  /* TODO */
}


/* TODO */


int Body_checksum(int id) {
  return static_cast<int>(host_Body_pos_x[id]*1000 + host_Body_pos_y[id]*2000
      + host_Body_vel_x[id]*3000 + host_Body_vel_y[id]*4000) % 123456;
}


__global__ void kernel_initialize_bodies(float* pos_x, float* pos_y,
                                         float* vel_x, float* vel_y,
                                         float* mass, float* force_x,
                                         float* force_y) {
  dev_Body_pos_x = pos_x;
  dev_Body_pos_y = pos_y;
  dev_Body_vel_x = vel_x;
  dev_Body_vel_y = vel_y;
  dev_Body_mass = mass;
  dev_Body_force_x = force_x;
  dev_Body_force_y = force_y;

  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kNumBodies; i += blockDim.x * gridDim.x) {
    // Initialize random state.
    curandState rand_state;
    curand_init(kSeed, i, 0, &rand_state);

    // Create new Body object.
    new_Body(/*id=*/ i,
             /*pos_x=*/ 2 * curand_uniform(&rand_state) - 1,
             /*pos_y=*/ 2 * curand_uniform(&rand_state) - 1,
             /*vel_x=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
             /*vel_y=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
             /*mass=*/ (curand_uniform(&rand_state)/2 + 0.5) * kMaxMass);
  }
}


__global__ void kernel_compute_force() {
  /* TODO */
}


__global__ void kernel_update() {
  /* TODO */
}


// Compute one step of the simulation.
void step_simulation() {
  kernel_compute_force<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_update<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


void run_benchmark() {
  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kBenchmarkIterations; ++i) {
    step_simulation();
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
      .count();

  printf("Time: %lu ms\n", millis);
}

int checksum() {
  int result = 0;

  for (int i = 0; i < kNumBodies; ++i) {
    result += Body_checksum(i);
  }

  return result;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s mode\n\nmode 1: Benchmark\n",
           argv[0]);
    return 1;
  }

  int mode = atoi(argv[1]);

  // Allocate and create Body objects.
  cudaMallocManaged(&host_Body_pos_x, sizeof(float)*kNumBodies);
  cudaMallocManaged(&host_Body_pos_y, sizeof(float)*kNumBodies);
  cudaMallocManaged(&host_Body_vel_x, sizeof(float)*kNumBodies);
  cudaMallocManaged(&host_Body_vel_y, sizeof(float)*kNumBodies);
  cudaMallocManaged(&host_Body_mass, sizeof(float)*kNumBodies);
  cudaMallocManaged(&host_Body_force_x, sizeof(float)*kNumBodies);
  cudaMallocManaged(&host_Body_force_y, sizeof(float)*kNumBodies);

  kernel_initialize_bodies<<<128, 128>>>(host_Body_pos_x, host_Body_pos_y,
                                         host_Body_vel_x, host_Body_vel_y,
                                         host_Body_mass, host_Body_force_x,
                                         host_Body_force_y);
  gpuErrchk(cudaDeviceSynchronize());

  if (mode == 1) {
    run_benchmark();
    printf("Checksum: %i\n", checksum());
  } else {
    printf("Invalid mode.\n");
    return 1;
  }

  cudaFree(host_Body_pos_x);
  cudaFree(host_Body_pos_y);
  cudaFree(host_Body_vel_x);
  cudaFree(host_Body_vel_y);
  cudaFree(host_Body_mass);
  cudaFree(host_Body_force_x);
  cudaFree(host_Body_force_y);

  return 0;
}

