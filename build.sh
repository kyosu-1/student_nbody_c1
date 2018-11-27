#!/bin/sh
/usr/local/cuda/bin/nvcc -DNDEBUG -std=c++11 nbody.cu rendering.cu -lSDL2 -o nbody -arch compute_61
/usr/local/cuda/bin/nvcc -DNDEBUG -std=c++11 nbody_soa.cu -o nbody_soa -arch compute_61
/usr/local/cuda/bin/nvcc -DNDEBUG -std=c++11 nbody_soaalloc.cu rendering_soa.cu -I. -Ilib/soa-alloc -o nbody_soaalloc -arch compute_61 --expt-relaxed-constexpr -maxrregcount=64 -lSDL2
