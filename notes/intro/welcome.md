# High Performance Computing Introduction
## Why HPC?
Single CPU performance is plateauing, we need to use parallel computing instead of serial computing.

## Why HPC is hard
Parallel computing can be done with shared memory or in a cluster configuration, or a hybrid of each (eg, Niagara and Graham). Message Passing (eg. OpenMP) and Non-Uniform Memory Access (NUMA) must be handeled.

The most efficient serial algorithm is often not the Num. Proc. -> 1 limit of the most efficient parallel algorithm.

## HPC Lessons

1. Always keep throughput in mind. Sometimes running more jobs each with fewer processors is better.

2. The optimal serial algorithm may be very structurally different than the optimal parallel algorithm

3. Parallel algorithm design is about finding as much concurrency as possible and maximizing locality.

4. The best approach to parallelizing your problem will depend on the details of the problem **and** the hardware.
