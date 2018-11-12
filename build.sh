#!/bin/sh
/usr/local/cuda/bin/nvcc -std=c++11 nbody.cu rendering.cu -lSDL2
