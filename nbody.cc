#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <new>

#include "nbody.h"
#include "rendering.h"

// Simulation parameters.
static const int kSeed = 42;
static const float kTimeInterval = 0.5;
static const int kBenchmarkIterations = 10000;

// Physical constants.
static const float kGravityConstant = 6.673e-11;   // gravitational constant

// Array containing all Body objects on device.
Body* host_bodies;


float random_float(float a, float b) {
  float random = ((float) rand()) / (float) RAND_MAX;
  return a + random * (b - a);
}


Body::Body(float pos_x, float pos_y,
           float vel_x, float vel_y, float mass) {
  /* TODO */
}


void Body::compute_force() {
  /* TODO */

  // Bodies should bounce off the wall when they go out of range.
  // Range: [-1, -1] to [1, 1]
}


void Body::update(float dt) {
  /* TODO */
}


void kernel_initialize_bodies() {
  srand(kSeed);

  for (int i = 0; i < kNumBodies; ++i) {
    // Create new Body object with placement-new.
    new(host_bodies + i) Body(/*pos_x=*/ random_float(-1.0, 1.0f),
                              /*pos_y=*/ random_float(-1.0, 1.0f),
                              /*vel_x=*/ random_float(-0.5, 0.5f) / 1000,
                              /*vel_y=*/ random_float(-0.5, 0.5f) / 1000,
                              /*mass=*/ random_float(0.5, 1.0f) * kMaxMass);
  }
}


void kernel_compute_force() {
  for (int i = 0; i < kNumBodies; ++i) {
    host_bodies[i].compute_force();
  }
}


void kernel_update() {
  for (int i = 0; i < kNumBodies; ++i) {
    host_bodies[i].update(kTimeInterval);
  }
}


// Compute one step of the simulation.
void step_simulation() {
  kernel_compute_force();
  kernel_update();
}


void run_interactive() {
  init_renderer();

  // Run simulation until user closes the window.
  do {
    // Compute one step.
    step_simulation();
  } while (draw(host_bodies));

  close_renderer();  
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

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s mode\n\nmode 0: Interactive mode\nmode 1: Benchmark\n",
           argv[0]);
    return 1;
  }

  int mode = atoi(argv[1]);

  // Allocate and create Body objects.
  host_bodies = new Body[kNumBodies];
  kernel_initialize_bodies();

  if (mode == 0) {
    run_interactive();
  } else if (mode == 1) {
    run_benchmark();
  } else {
    printf("Invalid mode.\n");
    return 1;
  }

  delete[] host_bodies;
  return 0;
}
