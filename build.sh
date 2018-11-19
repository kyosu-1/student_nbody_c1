#!/bin/sh
/usr/local/cuda/bin/nvcc -std=c++11 nbody.cu rendering.cu -lSDL2 -o nbody
/usr/local/cuda/bin/nvcc -std=c++11 nbody_soa.cu -o nbody_soa
