#include <curand_kernel.h>

#include "dynasoar.h"

class Body;

using AllocatorT = SoaAllocator<64*64*64*64, Body>;

class Body : public AllocatorT::Base {
  public:
    declare_field_types(Body, Body*, float, float, float, float, float, float)
  
  private:
    Field<Body, 0> merge_into_;
    Field<Body, 1> pos_x_;
    Field<Body, 2> pos_y_;
    Field<Body, 3> vel_x_;
    Field<Body, 4> vel_y_;
    Field<Body, 5> mass_;
    Field<Body, 6> force_x_;
    Field<Body, 7> force_y_;


  public:
    __device__ Body(float pos_x, float pos_y, float vel_x, float vel_y, float mass);

    __device__ Body(int idx);

    __device__ void compute_force();

    __device__ void apply_force(Body *other);

    __device__ void update();

    __device__ void check_merge_into_this(Body* other);

    __device__ void initialize_merge();

    __device__ void prepare_merge();

    __device__ void update_merge();

    __device__ void delete_merged();

    void add_checksum();

    __device__ __host__ float pos_x() const { return pos_x_; }
    __device__ __host__ float pos_y() const { return pos_y_; }
    __device__ __host__ float mass() const { return mass_; }


};