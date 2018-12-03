# Problem 3: Rewrite n-body Simulation with SoaAlloc
SoaAlloc is a CUDA library for programming with Structure of Arrays (SOA) data layout that we are developing at our research group. Programs that use SoaAlloc get the benefit of the SOA data layout (`nbody_soa.cu`) and the notation of a standard AOS layout (`nbody.cu`). Your task is to rewrite the n-body simulation with SoaAlloc.


0. Copy your solution from Problem 2 to `nbody.cu` and `nbody_soa.cu`. Note that this file has changed slightly (checksum computation).
1. Get familar with SoaAlloc. Read the [README](https://github.com/prg-titech/soa-alloc/blob/master/README.md) file.
2. Fill in the `TODO` parts in `nbody_soaalloc.cu`.
3. Run both `nbody`, `nbody_soa` and `nbody_soaalloc` in benchmark mode. They should produce the same checksum. Which version is faster?
