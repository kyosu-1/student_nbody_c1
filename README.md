# Problem 0: Implement CPU n-body Simulation
This repository contains a skeleton of an n-body simulation written in C++. An n-body simulation simulates the movement of asteroids or planets (bodies) by calculating the [gravitational force](https://en.wikipedia.org/wiki/Gravity#Newton's_theory_of_gravitation) between every pair of bodies.

```math
F = G * m1 * m2 / (r * r)
```

```math
F = m * a
```

![nbody animation](https://github.com/prg-titech/student_nbody_c1/raw/master/nbody.gif "nbody animation")

Your first task is to complete the missing parts of the implementation code in `nbody.cc`. There are two main functions in this skeleton. `Body::compute_force()` accumulates the total force that is exerted on a given body. `Body::update(dt)` computes the change in velocity and position of a given body based on the force. We compute these two parts separately to make sure that always the "old" position is used for calculating forces. This will become important once we switch to CUDA, such that the result is deterministic and always the same.

0. Set up your Linux programming environment. Ensure that a suitable C++ compiler, CUDA and libsdl2 are installed. If libsdl2 and your C++ compiler are installed correctly, you should be able to build and run the code in this repository. If you do not have access to a Linux machine, you can log into our lab machine.
1. Fill in the `TODO` parts in `nbody.h` and `nbody.cu`.
2. Until our next meeting, get familiar with CUDA. Read through and experiment with Nvidia's [CUDA tutorial](https://devblogs.nvidia.com/even-easier-introduction-cuda/).

## Login to Lab Machine
You need 2 terminal windows. In the first window, tunnel our lab machine's SSH port through the lab gateway, which is accessible from anywhere on the internet.

```
ssh  -N -L 3000:192.168.62.78:22 user_name@f9.is.titech.ac.jp -p 1104
```

Then you can connect to the server through your local machine.

```
ssh user_name@127.0.0.1 -p 3000 -Y
```

We will post the login credentials in the Slack. I suggest that you set up public key authentication to make the login more convenient.

If you use Linux, this should just work out of the box. If you use Mac, you have to install [XQuartz](https://www.xquartz.org/) first and run both commands through an XQuartz terminal. If you use Windows, you have to install an X server through [Cygwin/X](http://x.cygwin.com/) and use the Cygwin terminal.

## Hints

* The rendering frame rate with X window forwarding can be very slow. You can increase `kTimeInterval` to simulate a longer period of time with every step.
* Run the program in interactive mode during debugging. You will see a graphical rendering of the simulation. You are done with this task when the rendering "looks good".
* If you use Ubuntu, you can install libsdl2 as `libsdl2-dev` and `libsdl2-2.0-0`.
