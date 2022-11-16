#!/bin/sh
/usr/local/cuda/bin/nvcc -DNDEBUG -std=c++11 nbody.cu rendering.cu -lSDL2 -o nbody -arch compute_61
/usr/local/cuda/bin/nvcc -DNDEBUG -std=c++11 nbody_soa.cu -o nbody_soa -arch compute_61
#/usr/local/cuda/bin/nvcc -DNDEBUG -std=c++11 nbody_soaalloc.cu rendering_soa.cu -I. -Ilib/soa-alloc -o nbody_soaalloc -arch compute_61 --expt-relaxed-constexpr -maxrregcount=64 -lSDL2
/usr/local/cuda/bin/nvcc -Xcudafe "--diag_suppress=1427" "$@" -O3 -DNDEBUG -std=c++11 -lineinfo --expt-extended-lambda -gencode arch=compute_50,code=sm_50 -gencode arch=compute_61,code=sm_61 -maxrregcount=64 -Ilib/dynasoar -I. -Ilib/dynasoar/lib/cub nbody_dynasoar.cu rendering_dynasoar.cu -lSDL2 -o nbody_dynasoar