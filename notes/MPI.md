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

### mpi-intro
The first script is just a hello world script. In C:

```C
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {

        int rank, size;
        int ierr;

        ierr = MPI_Init(&argc, &argv);

        ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
        ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        printf("Hello, world from task %d of %d!\n",rank,size);

        MPI_Finalize();

        return ierr;
}
```

Compile using `mpicc hello-world.c -o hello-worldc`.

`mpicc` is a wrapper around compiling the program for multi-threading. Using the `--show-me` flag shows what it's really doing.

```bash
$ mpicc --show-me hello-world.c -o hello-worldc

gcc hello-world.c -o hello-worldc -I/scinet/niagara/software/2018a/opt/gcc-7.3.0/openmpi/3.1.0/include/openmpi
 -I/scinet/niagara/software/2018a/opt/gcc-7.3.0/openmpi/3.1.0/include/openmpi/opal/mca/hwloc/hwloc1117/hwloc/include -I/scinet/niagara/software/2018a/opt/gcc-7.3.0/openmpi/3.1.0/include/openmpi/opal/mca/event/libevent2022/libevent
 -I/scinet/niagara/software/2018a/opt/gcc-7.3.0/openmpi/3.1.0/include/openmpi/opal/mca/event/libevent2022/libevent/include
 -I/scinet/niagara/software/2018a/opt/gcc-7.3.0/openmpi/3.1.0/include -pthread -L/opt/slurm/lib64
 -L/scinet/niagara/mellanox/hpcx-2.1.0-ofed-4.3/hcoll/lib
 -L/scinet/niagara/mellanox/hpcx-2.1.0-ofed-4.3/mxm/lib
 -L/scinet/niagara/mellanox/hpcx-2.1.0-ofed-4.3/ucx/lib -Wl,-rpath -Wl,/opt/slurm/lib64 -Wl,-rpath
 -Wl,/scinet/niagara/mellanox/hpcx-2.1.0-ofed-4.3/hcoll/lib -Wl,-rpath
 -Wl,/scinet/niagara/mellanox/hpcx-2.1.0-ofed-4.3/mxm/lib -Wl,-rpath
 -Wl,/scinet/niagara/mellanox/hpcx-2.1.0-ofed-4.3/ucx/lib -Wl,-rpath
 -Wl,/scinet/niagara/software/2018a/opt/gcc-7.3.0/openmpi/3.1.0/lib -Wl,--enable-new-dtags
 -L/scinet/niagara/software/2018a/opt/gcc-7.3.0/openmpi/3.1.0/lib -lmpi
```

#### Explanation of `hello-world.c`
See the program above. Line-by-line:

 * `MPI_Init(&argc, &arv)`: MPI initialization
 * `MPI_Comm_size1`, `MPI_Comm_rank`: Communicator components (more later)
 * `MPI_Finalize`: Finalizes the MPI code, must be called last. (Sometimes omitted but this is bad style!)

Error handling: In C, the error code is returned as the return value of the function call. In Fortran, it's passed as an argument to the functions.

However, the way MPI usually handles errors is just to crash. So it's not really necessary to check the error code. This is an accepted practice.

The **rank** is essentially the only thing different between processes.

#### make
The Make program can also be ran in parallel! `make -j N` runs make in parallel on N cores.

## Communicators
MPI groups processes into communicators. Each communicator has some **size**, a number of tasks. Each tasks is assigned a **rank**, from `0..SIZE - 1`. Each task in the program belongs to the global communicator, `MPI_COMM_WORLD`.

The basic functions for using communicators are

 * `MPI_COMM_WORLD`: Global communicator
 * `MPI_Comm_rank(MPI_COMM_WORLD, &rank)`: get current tasks rank
 * `MPI_Comm_size(MPI_COMM_WORLD, &size)`: get communicator size

We don't have to stick with the global communicator. It's possible to have user defined communicators over the same tasks as the global, or break tasks into subgroups.

## Sending our first message.
Now that the basics are covered, we can write, compile, and run some code! The code for this section is in the `2_mpi/` directory.

To start, break down the `firstmessage.c` program into blocks

 * Header, including the `mpi` library
```C++
   #include <stdio.h>
   #include <mpi.h>
```

 * initialize variables, specify sendmessage and getmessage as char types.
```C
   int main(int argc, char **argv) {
       int rank, size, ierr;
       int sendto, recvfrom;  // task to send, recv from
       int ourtag=1;          // shared tag to label msgs
       char sendmessage[]="Hello";       // text to send
       char getmessage[6];            // text to recieve
       MPI_Status rstatus;       // MPI_Recv status info
       ...
     }
```

  Initialize Communicator, and set rank and size.
```C
       ierr = MPI_Init(&argc, &argv);
       ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
       ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
```
 * Check the rank, and send message if the rank is 0; receive the message if the rank is 1.
    * MPI_Ssend: Secure send; specify length of message, `MPI_CHAR` is the datatype, `ourtag` is the tag, and give communicator
    * MPI_Recf: Again specify length and type, tag, and communicator. rstatus is the receive status.

   ```C
     if (rank == 0) {
         sendto = 1;
         ierr = MPI_Ssend(sendmessage, 6, MPI_CHAR, sendto,
                          ourtag, MPI_COMM_WORLD);
         printf("%d: Sent message <%s>\n", rank, sendmessage);
       }
      else if (rank == 1) {
         recvfrom = 0;
         ierr = MPI_Recv(getmessage, 6, MPI_CHAR, recvfrom,
                         ourtag, MPI_COMM_WORLD, &rstatus);
         printf("%d: Got message <%s>\n", rank, getmessage);
       }
  ```


Compile and run the program:
```bash
mpicc firstmessage.c -o firstmessage
srun -np 2 firstmessage
```

## More complicated example

What if we want to do a similar thing as before, but with an arbitrary number of ranks. We want to keep sending to the right. What do we do at the end? We use `MPI_PROV_NULL` to send the message "nowhere".

  0 --> 1 --> 2 --> ... n -->

  secondmessage.c:
  ```C++
  #include <stdio.h>
  #include <mpi.h>

  int main(int argc, char **argv) {
      int rank, size, ierr;
      int left, right;
      int tag=1;
      double msgsent, msgrcvd;
      MPI_Status rstatus;

      ierr = MPI_Init(&argc, &argv);
      ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
      ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      left = rank - 1;
      if (left < 0) left = MPI_PROC_NULL;
      right = rank + 1;
      if (right == size) right = MPI_PROC_NULL;

      msgsent = rank*rank;
      msgrcvd = -999;

      ierr = MPI_Ssend(&msgsent, 1, MPI_DOUBLE, right,
                       tag, MPI_COMM_WORLD);
      ierr = MPI_Recv(&msgrcvd, 1, MPI_DOUBLE, left,
                       tag, MPI_COMM_WORLD, &rstatus);

      printf("%d: Sent %lf and got %lf\n",
                  rank, msgsent, msgrcvd);

      ierr = MPI_Finalize();
      return 0;
  }
```

What if we want periodic boundary conditions:
```
    0 --> 1 --> 2
    ^           |   
    |           v
    ------------
```
We expect we can just modify the program

```C++
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size, ierr;
    int left, right;
    int tag=1;
    double msgsent, msgrcvd;
    MPI_Status rstatus;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    left = rank - 1;
    if (left < 0) left = size-1;
    right = rank + 1;
    if (right == size) right = 0;

    msgsent = rank*rank;
    msgrcvd = -999;

    ierr = MPI_Ssend(&msgsent, 1, MPI_DOUBLE, right,
                     tag, MPI_COMM_WORLD);
    ierr = MPI_Recv(&msgrcvd, 1, MPI_DOUBLE, left,
                     tag, MPI_COMM_WORLD, &rstatus);

    printf("%d: Sent %lf and got %lf\n",
                rank, msgsent, msgrcvd);

    ierr = MPI_Finalize();
    return 0;
}
```

We compile and run this

```bash
make thirdmessagec
srun -n 4 ./thirdmessagec
```

And it sort of just hangs. Why?

This is called a **deadlock**. Each task is listening for a message, so they can't receive. This is unique to parallel processing; doesn't happen in serial. This is a big idea in MP.

> When you send a message, you need to make sure there is a process ready to receive the message.

How do we fix this without a completely new MPI routine?

One option is with even/odd ranks. Set the even ranks to send, and the odd ranks receive.

```C++
...
if (rank % 2 == 0) {
      ierr = MPI_Ssend(&msgsent, 1, MPI_DOUBLE, right,
                       tag, MPI_COMM_WORLD);
      ierr = MPI_Recv(&msgrcvd, 1, MPI_DOUBLE, left,
                       tag, MPI_COMM_WORLD, &rstatus);
  } else {
      ierr = MPI_Recv(&msgrcvd, 1, MPI_DOUBLE, left,
                       tag, MPI_COMM_WORLD, &rstatus);
      ierr = MPI_Ssend(&msgsent, 1, MPI_DOUBLE, right,
                       tag, MPI_COMM_WORLD);
...
```

This runs:
```bash
$ make fourthmessagec
$ srun -n 4 ./fourthmessagec
0: Sent 0.000000 and got 9.000000
1: Sent 1.000000 and got 0.000000
2: Sent 4.000000 and got 1.000000
3: Sent 9.000000 and got 4.000000
```

This is such a common pattern in MPI, that openMP implements a function for this:

```C++
    err = MPI_sendrecv(sendptr, count, MPI_TYPE, destination, tag,
                      recvptr, count, MPI_TYPE, source, tag, Communicator, MPI_status)
```

In our code we add

```C++
ierr = MPI_Sendrecv(&msgsent, 1, MPI_DOUBLE, right, tag,
                    &msgrcvd, 1, MPI_DOUBLE, left,  tag,
                    MPI_COMM_WORLD, &rstatus);
```

The final version of the message is

```C++
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size, ierr;
    int left, right;
    int tag=1;
    double msgsent, msgrcvd;
    MPI_Status rstatus;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    left = rank - 1;
    if (left < 0) left = size-1;
    right = rank + 1;
    if (right == size) right = 0;

    msgsent = rank*rank;
    msgrcvd = -999;

    ierr = MPI_Sendrecv(&msgsent, 1, MPI_DOUBLE, right, tag,
                        &msgrcvd, 1, MPI_DOUBLE, left,  tag,
                        MPI_COMM_WORLD, &rstatus);

    printf("%d: Sent %lf and got %lf\n",
                rank, msgsent, msgrcvd);

    ierr = MPI_Finalize();
    return 0;
}
```

## Reductions
Consider a problem structure where we need to collect the parallel computation results for some final analysis. For instance, consider taking the min, mean, and max of a dataset. Below is a simple serial solution for random data.

 ```C++
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const int nx=1500;
    float *dat;
    int i;
    float datamin, datamax, datamean;

    /*
     * generate random data
     */

    dat = (float *)malloc(nx * sizeof(float));
    srand(0);
    for (i=0;i<nx;i++) {
        dat[i] = 2*((float)rand()/RAND_MAX)-1.;
    }

    /*
     * find min/mean/max
     */

    datamin = 1e+19;
    datamax =-1e+19;
    datamean = 0;


    for (i=0;i<nx;i++) {
        if (dat[i] < datamin) datamin=dat[i];
        if (dat[i] > datamax) datamax=dat[i];
        datamean += dat[i];
    }
    datamean /= nx;
    free(dat);

    printf("Min/mean/max = %f,%f,%f\n", datamin,datamean,datamax);

    return 0;
}
```

How can we implement this in parallel? The naive application is to find the min/mean/max of each process, and send this to some `root` processor to reduce the results. This is an easy adaptation of the serial program.

```C++
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const int nx=1500;
    float *dat;
    int i;
    float datamin, datamax, datamean;
    float minmeanmax[3];
    float globminmeanmax[3];
    int ierr;
    int rank, size;
    int tag=1;
    int masterproc=0;
    MPI_Status status;


    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    /*
     * generate random data
     */

    dat = (float *)malloc(nx * sizeof(float));
    srand(rank*rank);
    for (i=0;i<nx;i++) {
        dat[i] = 2*((float)rand()/RAND_MAX)-1.;
    }

    /*
     * find min/mean/max
     */

    datamin = 1e+19;
    datamax =-1e+19;
    datamean = 0;

    for (i=0;i<nx;i++) {
        if (dat[i] < datamin) datamin=dat[i];
        if (dat[i] > datamax) datamax=dat[i];
        datamean += dat[i];
    }
    datamean /= nx;
    free(dat);
    printf("Min/mean/max = %f,%f,%f\n", datamin,datamean,datamax);

    minmeanmax[0] = datamin;
    minmeanmax[2] = datamax;
    minmeanmax[1] = datamean;

    if (rank != masterproc) {
       ierr = MPI_Ssend(minmeanmax,3,MPI_FLOAT,masterproc,tag,MPI_COMM_WORLD);
    } else {
        globminmeanmax[0] = datamin;
        globminmeanmax[2] = datamax;
        globminmeanmax[1] = datamean;
        for (i=1;i<size-1;i++) {
            ierr = MPI_Recv(minmeanmax,3,MPI_FLOAT,MPI_ANY_SOURCE,tag,MPI_COMM_WORLD,
                     &status);

            globminmeanmax[1] += minmeanmax[1];

            if (minmeanmax[0] < globminmeanmax[0])
                globminmeanmax[0] = minmeanmax[0];

            if (minmeanmax[2] > globminmeanmax[2])
                globminmeanmax[2] = minmeanmax[2];

        }
        globminmeanmax[1] /= size;
        printf("Min/mean/max = %f,%f,%f\n", globminmeanmax[0],
               globminmeanmax[1],globminmeanmax[2]);
    }

    ierr = MPI_Finalize();

    return 0;
}
```

However, consider that for P processors we send (P-1) messages in serial. Passing messages is expensive, so this is not a good design pattern.

We could program ourselves to pass messages pairwise, then pairwise between remaining processes, etc. The MPI library implements this with the functions `MPI_Allreduce` and `MPI_reduce`:

```C++
err = MPI_Allreduce(sendptr, rcvptr, count, MPI_TYPE, MPI_Op, Communicator);
err = MPI_reduce(sendbuf, rcvbuf, count, MPI_TYPE, MPI_Op, root, Communicator);
```

 * `sendptr/rcvptr`: pointers to buffers for send/receive
 * `count`: number of elements in ptrs
 * `MPI_TYPE`: type of message (eg. `MPI_DOUBLE`)
 * `MPI_Op`: one of `MPI_SUM`, `MPI_PROD`, `MPI_MIN`, `MPI_MAX`.
 * Communicator: `MPI_COMM_WORLD` or user created
 * `MPI_Allreduce` sends result back to everyone; `MPI_reduce` sends result to `root`.

 With these functions, the program reduces to the following

```C++
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const int nx=1500;
    float *dat;
    int i;
    float datamin, datamax, datamean;
    float globalmin, globalmax, globalmean;
    int ierr;
    int rank, size;
    int tag=1;
    int masterproc=0;
    MPI_Status status;


    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    /*
     * generate random data
     */

    dat = (float *)malloc(nx * sizeof(float));
    srand(rank*rank);
    for (i=0;i<nx;i++) {
        dat[i] = 2*((float)rand()/RAND_MAX)-1.;
    }

    /*
     * find min/mean/max
     */

    datamin = 1e+19;
    datamax =-1e+19;
    datamean = 0;

    for (i=0;i<nx;i++) {
        if (dat[i] < datamin) datamin=dat[i];
        if (dat[i] > datamax) datamax=dat[i];
        datamean += dat[i];
    }
    datamean /= nx;
    free(dat);

    ierr = MPI_Allreduce(&datamin, &globalmin, 1, MPI_FLOAT,
                         MPI_MIN, MPI_COMM_WORLD);
    /*
     * to just sent to rank 0:
     * MPI_Reduce(datamin, globalmin, 1, MPI_FLOAT, &
     *                MPI_MIN, 0, MPI_COMM_WORLD)
     */
    ierr = MPI_Allreduce(&datamax, &globalmax, 1, MPI_FLOAT,
                         MPI_MAX, MPI_COMM_WORLD);
    ierr = MPI_Allreduce(&datamean, &globalmean, 1, MPI_FLOAT,
                         MPI_SUM, MPI_COMM_WORLD);
    globalmean /= size;

    printf("Min/mean/max = %f,%f,%f\n", datamin,datamean,datamax);

    if (rank == 0) {
        printf("Global Min/mean/max = %f,%f,%f\n",
                globalmin, globalmean, globalmax);
    }

    ierr = MPI_Finalize();

    return 0;
}
```

The reduce functions are examples of MPI *collectives*. Some other collectives are:

 * Broadcast
 * Scatter
 * Gather

### Example: scatter and gather
Consider the following program that sends data to all the processes

```C++
#include "mpi.h"
#include <stdio.h>
#define SIZE 4

int main(int argc, char* argv[]) {


  int numtasks, rank, sendcount, recvcount, source;
  float sendbuf[SIZE][SIZE] = {
    {1.0, 2.0, 3.0, 4.0},
    {5.0, 6.0, 7.0, 8.0},
    {9.0, 10.0, 11.0, 12.0},
    {13.0, 14.0, 15.0, 16.0}  };
  float recvbuf[SIZE];
  float newbuf[SIZE][SIZE];

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  if (numtasks == SIZE) {
      source = 1;
      sendcount = SIZE;
      recvcount = SIZE;
      MPI_Scatter(sendbuf,sendcount,MPI_FLOAT,recvbuf,recvcount,
                  MPI_FLOAT,source,MPI_COMM_WORLD);
      }
  else
  printf("Must specify %d processors. Terminating.\n",SIZE);
  MPI_Finalize();
  }
```

**Exercise**: modify the program to use `MPI_Gather` to recreate the initial data `sendbuf` from the scattered data.

*Solution*: Include the definition `float [SIZE][SIZE] newbuf` and add the following code to the end of the `if (numtasks == SIZE)` block

```C
      MPI_Gather(recvbuf, SIZE, MPI_FLOAT, newbuf, SIZE,
                  MPI_FLOAT, source, MPI_COMM_WORLD)

      if ( rank == source) {
        for (int i = 0;i<SIZE; i++)
          for (int j=0;j<SIZE; j++)
            printf("%f ", newbuf[i][j]);
          printf("\n")
        }
```
