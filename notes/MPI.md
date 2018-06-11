# Parallel Programming Clusters with MPI
*June 11-12, 2018. University of Toronto, Wilson Hall. Ramses van Zon*.

One paradigm for parallel computing is **distributed memory**. From a hardware perspective, simply connect multiple computers together with a network. We think of the communication between the CPUs as a **message passing interface** (MPI).

For programming languages, we are looking at high performance compiled programming languages: C, C++, Fortran.

## Message Passing Interface (MPI)
MPI itself is a standard library interface for message passing, ratified by the MPI Forum.

[OpenMPI](https://www.open-mpi.org/) is an open source library, and is one option on Niagara (and Graham) for implementing MPI. On Niagara, load with `module load gcc openmpi`.

[MPICH2](https://www.mpich.org/) is another option; again load with `module load intelmpi`.

MPI is a library, so you need to include its header file, eg:

```C++
#include <stdio.h>
#include <mpi.h>
...
```

The MPI Library is huge (>200 functions), but not as many concepts. Most of message passing can be understood with just a few functions.
