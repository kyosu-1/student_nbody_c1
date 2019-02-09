#include <chrono>
#include <curand_kernel.h>

#include "allocator/soa_allocator.h"
#include "allocator/soa_base.h"
#include "allocator/allocator_handle.h"

#include "configuration.h"
#include "rendering_soa.h"

#define OPTION_DRAW true

// Pre-declare all classes.
class Body;

using AllocatorT = SoaAllocator<64*64*64*64, Body>;

class Body : public SoaBase<AllocatorT> {
 public:
  using FieldTypes = std::tuple<float, float, float, float, float, float, float>;

  SoaField<Body, 0> pos_x_;
  SoaField<Body, 1> pos_y_;
  SoaField<Body, 2> vel_x_;
  SoaField<Body, 3> vel_y_;
  SoaField<Body, 4> mass_;
  SoaField<Body, 5> force_x_;
  SoaField<Body, 6> force_y_;


  __device__ Body(float pos_x, float pos_y, float vel_x, float vel_y, float mass);

  __device__ void compute_force();

  __device__ void apply_force(Body* other);

  __device__ void update();

  // Only for rendering purposes.
  __device__ void add_to_draw_array();
};

// Allocator handles.
AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;


// Helper variables for rendering and checksum computation.
__device__ int draw_counter = 0;
__device__ float Body_pos_x[kNumBodies];
__device__ float Body_pos_y[kNumBodies];
__device__ float Body_vel_x[kNumBodies];
__device__ float Body_vel_y[kNumBodies];
__device__ float Body_mass[kNumBodies];
float host_Body_pos_x[kNumBodies];
float host_Body_pos_y[kNumBodies];
float host_Body_vel_x[kNumBodies];
float host_Body_vel_y[kNumBodies];
float host_Body_mass[kNumBodies];

__device__ Body::Body(float pos_x, float pos_y,
                      float vel_x, float vel_y, float mass)
    : pos_x_(pos_x), pos_y_(pos_y),
      vel_x_(vel_x), vel_y_(vel_y), mass_(mass) {}


__device__ void Body::compute_force() {
  force_x_ = 0;
  force_y_ = 0;
  device_allocator->template device_do<Body>(&Body::apply_force, this);
  /*for (int i = 0; i < kNumBodies; ++i){
    this.apply_force(dev_bodies + i)*/
}



__device__ void Body::apply_force(Body* other) {
  // Update `other`.
  if(this != other){
    float m1 = this->mass_;
    float dx = this->pos_x_ - other->pos_x_;
    float dy = this->pos_y_ - other->pos_y_;
    float r = sqrt(dx * dx + dy * dy);
    other->force_x_ += kGravityConstant * m1 * other->mass_ / (r * r * r) * dx;
    other->force_y_ += kGravityConstant * m1 * other->mass_ / (r * r * r) * dy;
  }
}


__device__ void Body::update() {
  vel_x_ += force_x_ / mass_ * kTimeInterval;
  vel_y_ += force_y_ / mass_ * kTimeInterval;
  pos_x_ += vel_x_ * kTimeInterval;
  pos_y_ += vel_y_ * kTimeInterval;
  if (abs(pos_x_) > 1) {
    vel_x_ = - vel_x_;
  }
  if (abs(pos_y_) > 1) {
    vel_y_ = - vel_y_;
  }
}


__device__ void Body::add_to_draw_array() {
  int idx = atomicAdd(&draw_counter, 1);
  Body_pos_x[idx] = pos_x_;
  Body_pos_y[idx] = pos_y_;
  Body_vel_x[idx] = vel_x_;
  Body_vel_y[idx] = vel_y_;
  Body_mass[idx] = mass_;
}


__global__ void kernel_initialize_bodies() {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  curandState rand_state;
  curand_init(kSeed, tid, 0, &rand_state);

  for (int i = tid; i < kNumBodies; i += blockDim.x * gridDim.x) {


    // Initialize random state.
    curandState rand_state;
    curand_init(kSeed, i, 0, &rand_state);

    // Create new Body object with placement-new.

    device_allocator->make_new<Body>(/*pos_x=*/ 2 * curand_uniform(&rand_state) - 1,
                             /*pos_y=*/ 2 * curand_uniform(&rand_state) - 1,
                             /*vel_x=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
                             /*vel_y=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
                             /*mass=*/ (curand_uniform(&rand_state)/2 + 0.5)
                                           * kMaxMass);
  }
}


__global__ void kernel_reset_draw_counters() {
  draw_counter = 0;
}


void transfer_data() {
  // Extract data from SoaAlloc data structure.
  kernel_reset_draw_counters<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  allocator_handle->parallel_do<Body, &Body::add_to_draw_array>();
  gpuErrchk(cudaDeviceSynchronize());

  // Copy data to host.
  cudaMemcpyFromSymbol(host_Body_pos_x, Body_pos_x,
                       sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(host_Body_pos_y, Body_pos_y,
  	                   sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(host_Body_vel_x, Body_vel_x,
  	                   sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(host_Body_vel_y, Body_vel_y,
  	                   sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(host_Body_mass, Body_mass, sizeof(float)*kNumBodies, 0,
                       cudaMemcpyDeviceToHost);
}


int checksum() {
  transfer_data();
  int result = 0;

  for (int i = 0; i < kNumBodies; ++i) {
  	int Body_checksum = static_cast<int>((host_Body_pos_x[i]*1000 + host_Body_pos_y[i]*2000
                        + host_Body_vel_x[i]*3000 + host_Body_vel_y[i]*4000)) % 123456;
    result += Body_checksum;
  }

  return result;
}


void run_interactive() {
  init_renderer();

  while (true) {
    allocator_handle->parallel_do<Body, &Body::compute_force>();
    allocator_handle->parallel_do<Body, &Body::update>();

    // Transfer and render.
    transfer_data();
    draw(host_Body_pos_x, host_Body_pos_y, host_Body_mass);
  }

  close_renderer();
}


void run_benchmark() {
  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kBenchmarkIterations; ++i) {
    allocator_handle->parallel_do<Body, &Body::compute_force>();
    allocator_handle->parallel_do<Body, &Body::update>();
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
      .count();

  printf("Time: %lu ms\n", millis);
}


int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s mode\n\nmode 0: Interactive mode\nmode 1: Benchmark\n",
           argv[0]);
    return 1;
  }

  int mode = atoi(argv[1]);

  AllocatorT::DBG_print_stats();

  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  // Allocate and create Body objects.
  kernel_initialize_bodies<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  if (mode == 0) {
    run_interactive();
  } else if (mode == 1) {
    run_benchmark();
    printf("Checksum: %i\n", checksum());
  } else {
    printf("Invalid mode.\n");
    return 1;
  }

  return 0;
}
