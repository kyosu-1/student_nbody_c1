# Problem 1: Implement n-body Simulation
This repository contains a skeleton of an n-body simulation written in CUDA. An n-body simulation simulates the movement of asteroids or planets (bodies) by calculating the [gravitational force](https://en.wikipedia.org/wiki/Gravity#Newton's_theory_of_gravitation) between every pair of bodies.

```math
F = G * m1 * m2 / (r * r)
```

In CUDA, an n-body simulation consists of 2 kernels. The first kernel accumulates the total force that is exerted on bodies. The second kernel computes the change in velocity and position of bodies based on the force. We divide the program into two kernels to make sure that always the "old" position is used for calculating forces, such that the result is deterministic and always the same.

0. Set up your Linux programming environment. Ensure that CUDA and libsdl2 are installed. If both are installed correctly, you should be able to build and run the code in this repository. If you do not have access to a Linux machine, you can log into our lab machine.
1. Get familiar with CUDA. We suggest the Nvidia's [CUDA tutorial](https://devblogs.nvidia.com/even-easier-introduction-cuda/).
2. Fill in the `TODO` parts in `nbody.h` and `nbody.cu`.


## Login to Lab Machine
You need 2 terminal windows. In the first window, tunnel our lab machine's SSH port through the lab gateway, which is accessible from anywhere on the internet.

```
ssh -L 3000:192.168.62.78:22 user_name@f9.is.titech.ac.jp -p 1104
```

Then you can connect to the server through your local machine.

```
ssh user_name@127.0.0.1 -p 3000 -Y
```

We will post the login credentials in the Slack. I suggest that you set up public key authentication to make the login more convenient.

If you use Linux, this should just work out of the box. If you use Mac, you have to install [XQuartz](https://www.xquartz.org/) first and run both commands through an XQuartz terminal. If you use Windows, you have to install an X server through [Cygwin/X](http://x.cygwin.com/) and use the Cygwin terminal.
