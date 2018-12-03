# Problem 4: n-body Simulation with Collisions
In this problem, you will extend the n-body simulation (SoaAlloc version) from problem 3 to support collisions. Such a simulation is used by astronomers to simulate the collision of galaxies. In essence, if two bodies in our simulation become too close, we merge them into a single body. The mass of the resulting body is the sum of the masses of the two original bodies and its velocity is determined according to the physical law of [perfectly inelastic collision](https://en.wikipedia.org/wiki/Inelastic_collision).

0. Copy your solution from Problem `nbody_soaalloc.cu`. Note that this file has changed slightly (rendering code).
1. Implement the collision behavior for the interactive mode. We do not care about the checksum/benchmark mode this time. You are done if the simulation rendering looks realistic.

## Hints
This assignment is more tricky than the previous ones with respect to concurrency and data race issues. For simplicity, consider only the case where no more than two bodies are merged at a time. How do we determine which bodies should be merged?

* Two bodies collide if their distance is less than `0.01`.
* Merging can be implemented as follows: Merge the smaller body into the bigger one. Update mass, position, velocity of the heavier body. Then, delete the lighter body from the simulation with `device_allocator->free<Body>(obj)`.
* Your solution will likely require multiple GPU kernels. One kernel `prepare_merge` will determine which body a given body should be merged with (if any). This is important because there may be multiple bodies within merging distance; you may choose an arbitrary body in such a case. Another kernel `perform_merge` will perform the actual merge.
* There are a few rare cases such as the following one. `n1 --> n2` means that `n1` should be merged into `n2` (`n2` is the heavier one). Consider the case that `a --> b` and `b --> c` in kernel `perform_merge`. This is tricky because both merges happen concurrently on the GPU: `b` is changed and, at the same time, `c` is updated based on `b`. It is up to you how to handle such cases. In this example, one valid approach would be to perform the merge `a --> b` only if `b` is not being merged into any other body at the same time.
* You will likely have to add some fields to class `Body`. Note that fields must be sorted by the size of their type in SoaAlloc. Otherwise, you will get a compile error.
* You will likely need to use an atomic compare-and-swap operation at some point. E.g., to decide which body should be merged into a given body if there are multiple. [Compare-and-swap](https://en.wikipedia.org/wiki/Compare-and-swap) is a fundamental operation in concurrent systems. Check the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) (`atomicCAS`) to see how it can be used from CUDA.
