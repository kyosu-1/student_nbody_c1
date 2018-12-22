# Problem 1: Implement n-body Simulation
This repository contains a skeleton of an n-body simulation written in CUDA. An n-body simulation simulates the movement of asteroids or planets (bodies) by calculating the [gravitational force](https://en.wikipedia.org/wiki/Gravity#Newton's_theory_of_gravitation) between every pair of bodies.

```math
F = G * m1 * m2 / (r * r)
```

```math
F = m * a
```

![nbody animation](https://github.com/prg-titech/student_nbody_c1/raw/master/nbody.gif "nbody animation")

In CUDA, an n-body simulation consists of 2 kernels. The first kernel accumulates the total force that is exerted on bodies. The second kernel computes the change in velocity and position of bodies based on the force. We divide the program into two kernels to make sure that always the "old" position is used for calculating forces, such that the result is deterministic and always the same.

0. Set up your Linux programming environment. Ensure that CUDA and libsdl2 are installed. If both are installed correctly, you should be able to build and run the code in this repository. The code was tested with the CUDA Toolkit 9.0, but any recent version should work. If you do not have access to a Linux machine, you can log into our lab machine.
1. Get familiar with CUDA. We suggest the Nvidia's [CUDA tutorial](https://devblogs.nvidia.com/even-easier-introduction-cuda/).
2. Fill in the `TODO` parts in `nbody.cu`. You are done when the rendering of the GPU nbody simulation looks similar to the CPU version.
3. Compare the performance of the CPU version and the GPU version. You may have to increase the number of bodies to see a difference. Make sure to run the program in benchmark mode.

