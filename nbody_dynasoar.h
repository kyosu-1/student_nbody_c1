#include <curand_kernel.h>

#include "dynasoar.h"

class Body;

using AllocatorT = SoaAllocator<64*64*64*64, Body>;

class Body : public AllocatorT::Base {
  public:
    declare_field_types(/* TODO */)
  
  private:
    /* TODO */

  public:
    __device__ Body(float pos_x, float pos_y, float vel_x, float vel_y, float mass);

    __device__ Body(int idx);

    __device__ void compute_force();

    __device__ void apply_force(Body *other);

    __device__ void update();

    void add_checksum();

    __device__ __host__ float pos_x() const { return pos_x_; }
    __device__ __host__ float pos_y() const { return pos_y_; }
    __device__ __host__ float mass() const { return mass_; }

};