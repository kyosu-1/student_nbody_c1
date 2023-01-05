#include <chrono>

#include "nbody_dynasoar.h"

#include "configuration.h"
#include "rendering_dynasoar.h"

AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;

float host_checksum;

__device__ Body::Body(float pos_x, float pos_y,
                      float vel_x, float vel_y, float mass)
    : pos_x_(pos_x), pos_y_(pos_y),
      vel_x_(vel_x), vel_y_(vel_y), mass_(mass) {}

__device__ void Body::compute_force() {
  force_x_ = 0.0f;
  force_y_ = 0.0f;
  // Body型のすべてのオブジェクトに対してapply_forceを実行する
  device_allocator->template device_do<Body>(&Body::apply_force, this);
}

__device__ void Body::apply_force(Body* other) {
  float dx = other->pos_x_ - pos_x_;
  float dy = other->pos_y_ - pos_y_;
  float r = sqrt(dx * dx + dy * dy);
  float force = kGravityConstant * mass_ * other->mass_ / (r * r);
  other -> force_x_ += force * dx / r;
  other -> force_y_ += force * dy / r;
}

__device__ void Body::update() {
  // update velocity
  vel_x_ += force_x_ * kTimeInterval / mass_;
  vel_y_ += force_y_ * kTimeInterval / mass_;
  // update position
  pos_x_ += vel_x_ * kTimeInterval;
  pos_y_ += vel_y_ * kTimeInterval;

  if (abs(pos_x_) > 1) {
    vel_x_ *= -1;
  }

  if (abs(pos_y_) > 1) {
    vel_y_ *= -1;
  }
}

void Body::add_checksum() {
  host_checksum += pos_x_ + pos_y_*2 + vel_x_*3 + vel_y_*4;
}

__device__ Body::Body(int idx) {
  curandState rand_state;
  curand_init(kSeed, idx, 0, &rand_state);

  pos_x_ = 2 * curand_uniform(&rand_state) - 1;
  pos_y_ = 2 * curand_uniform(&rand_state) - 1;
  vel_x_ = (curand_uniform(&rand_state) - 0.5) / 1000;
  vel_y_ = (curand_uniform(&rand_state) - 0.5) / 1000;
  mass_ = (curand_uniform(&rand_state)/2 + 0.5)* kMaxMass;
  force_x_ = 0.0f;
  force_y_ = 0.0f;
}

void step_simulation() {
  allocator_handle->parallel_do<Body, &Body::compute_force>();
  allocator_handle->parallel_do<Body, &Body::update>();
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

bool render_frame() {
  init_frame();

  allocator_handle->template device_do<Body>([&](Body* body){
    draw_body(body->pos_x(), body->pos_y(), body->mass());
  });

  return show_frame();
}

void run_interactive() {
  init_renderer();

  do {
    step_simulation();
  } while (render_frame());

  close_renderer();
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s mode\n\nmode 0: Interactive mode\nmode 1: Benchmark\n",
           argv[0]);
    return 1;
  }

  int mode = atoi(argv[1]);

  allocator_handle = new AllocatorHandle<AllocatorT>(/*unified_memory=*/ true);
  AllocatorT *dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);
  
  // Initialize bodies
  for (int i = 0; i < kNumBodies; ++i) {
    allocator_handle->template device_new<Body>(i);
  }

  if (mode == 0) {
    run_interactive();
  } else if (mode == 1) {
    run_benchmark();
    allocator_handle->template device_do<Body>(&Body::add_checksum);
    printf("Checksum: %i\n", static_cast<int>(host_checksum));
  } else {
    printf("Invalid mode.\n");
    return 1;
  }

  return 0;
}