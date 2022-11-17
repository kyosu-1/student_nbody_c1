# Problem 2: Convert n-body Simulation to SOA Data Layout
In this problem, you will implement the same n-body simulation as in Problem 1, but in [SOA data layout](https://en.wikipedia.org/wiki/AOS_and_SOA). In the Problem 1, we defined an array `dev_bodies` of type `Body[]`. In SOA data layout, we define a separate array for every field of `Body`. E.g., there will arrays `Body_pos_x` and `Body_pos_y` of size `float[]`.

The SOA data layout usually achieves better performance on GPUs. The reason for that is that the GPU can access global memory more efficiently ([vector](https://en.wikipedia.org/wiki/Vector_processor) loads).

0. Copy your solution from Problem 1 to `nbody.cu`. Note that this file has changed slightly. E.g., there is now a function for computing a checksum in benchmark mode.
1. Ensure that you can build the project. There should be two executables `nbody` and `nbody_soa`.
2. Fill in thr `TODO` parts in `nbody_soa.cu`.
3. Run both `nbody` and `nbody_soa` in benchmark mode. They should produce the same checksum. Which version is faster?
