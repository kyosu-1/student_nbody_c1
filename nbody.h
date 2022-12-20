#ifndef NBODY_H
#define NBODY_H

// Simulation parameters.
// static const int kNumBodies = 25;
// static const float kMaxMass = 1000;

class Body {
 public:
  // Default constructor required.
  __device__ Body() {}

  // Constructor: Create new Body object.
  __device__ Body(float pos_x, float pos_y,
                  float vel_x, float vel_y, float mass);

  __device__ void compute_force();

  __device__ void update(float dt);

  void draw();

 private:
  float pos_x_;
  float pos_y_;
  float vel_x_;
  float vel_y_;
  float mass_;

  // Temporary variables.
  float force_x_;
  float force_y_;
};

#endif  // NBODY_H
