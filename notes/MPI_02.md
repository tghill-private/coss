# Parallel Programming Clusters with MPI
*June 12, 2018. University of Toronto, Wilson Hall. Ramses van Zon*.


## Scientific MPI Example
Intuitively, there are many different types of scientific problems that can inherently take advantage of MPI. Often this involves discretizing the continuous equations of the model on a grid.

### Discretizing Derivatives
Use a second-order centered difference method to discretize the diffusion equation

![DiffusionEq](https://latex.codecogs.com/gif.latex?\frac{\partial&space;T}{\partial&space;t}&space;=&space;K&space;\frac{\partial^2&space;T}{\partial&space;x^2})

And

![DiscretizeDeriv](https://latex.codecogs.com/gif.latex?\frac{\partial^2&space;T}{\partial&space;x^2}&space;\approx&space;\frac{T_{i&plus;1}&space;-&space;2T_{i}&space;&plus;&space;T_{i-1}}{\Delta&space;x^2})

This is written in 1-D, but we could discretize the equation in higher dimensions.

What about boundaries? We use *Guard Cells*. Pad the domain with guard cells so the stencil works for the first and last point in the domain. We fill the guard cells with values such that the boundary conditions are met.

The discretized derivative in the code looks like

```C
for (i=1; i<totpoints+1; i++) {
    temperature[new][i] = temperature[old][i] + dt*kappa/(dx*dx) *
        (temperature[old][i+1] - 2. * temperature[old][i] + temperature[old][i-1]) ;
}
```

### Serial code
Take a look at the serial code here:

```C
  #include <stdio.h>
  #include <stdlib.h>
  #include <math.h>
  #include "cpgplot.h"

  int main(int argc, char **argv) {
      // simulation parameters
      const int totpoints=1000;
      const float xleft = -12., xright = +12.;
      const float kappa = 1.;
      const int nsteps=100000;
      const int plotsteps=50;

      // data structures
      float * x;
      float ** temperature;
      float * theory;

      // parameters of the original temperature distribution
      const float ao=1., sigmao=1.;
      float a, sigma;

      float fixedlefttemp, fixedrighttemp;

      int old, new;
      int step, i;
      int red, grey,white;
      float time;
      float dt, dx;
      float error;

      // set parameters

      dx = (xright-xleft)/(totpoints-1);
      dt = dx*dx * kappa/10.;

       // allocate data, including ghost cells: old and new timestep
       // theory doesn't need ghost cells, but we include it for simplicity

      theory = (float * )malloc((totpoints+2) * sizeof(float));
      x      = (float * )malloc((totpoints+2)*  sizeof(float));
      temperature = (float ** )malloc(2 * sizeof(float *));
      temperature[0] = (float * )malloc((totpoints+2) * sizeof(float));
      temperature[1] = (float * )malloc((totpoints+2) * sizeof(float));
      old = 0;
      new = 1;

      // setup initial conditions

      time = 0.;
      for (i=0; i<totpoints+2; i++) {
          x[i] = xleft + (i-1+0.5) * dx;
          temperature[old][i] = ao*exp(-(x[i] * x[i]) / (2. * sigmao * sigmao));
          theory[i]           = ao*exp(-(x[i] * x[i]) / (2.* sigmao * sigmao));
      }
      fixedlefttemp = ao*exp(-(xleft-dx) * (xleft-dx) / (2. * sigmao * sigmao));
      fixedrighttemp= ao*exp(-(xright+dx) * (xright+dx)/(2. * sigmao*sigmao));

  #ifdef PGPLOT
      cpgbeg(0, "/xwindow", 1, 1);
      cpgask(0);
      cpgenv(xleft, xright, 0., 1.5*ao, 0, 0);
      cpglab("x", "Temperature", "Diffusion Test");
      red = 2;  cpgscr(red,1.,0.,0.);
      grey = 3; cpgscr(grey,.2,.2,.2);
      white=4;cpgscr(white,1.0,1.0,1.0);

      cpgsls(1); cpgslw(1); cpgsci(grey);
      cpgline(totpoints+2, x, theory);
      cpgsls(2); cpgslw(3); cpgsci(red);
      cpgline(totpoints+2, x, temperature[old]);
  #endif

      // evolve

      for (step=0; step < nsteps; step++) {
          // boundary conditions: keep endpoint temperatures fixed

          temperature[old][0] = fixedlefttemp;
          temperature[old][totpoints+1] = fixedrighttemp;

          for (i=1; i<totpoints+1; i++) {
              temperature[new][i] = temperature[old][i] + dt*kappa/(dx*dx) *
                  (temperature[old][i+1] - 2. * temperature[old][i] +
                   temperature[old][i-1]) ;
          }


          time += dt;
  #ifdef PGPLOT
          if (step % plotsteps == 0) {
              cpgbbuf();
              cpgeras();
              cpgsls(2); cpgslw(12); cpgsci(red);
              cpgline(totpoints+2, x, temperature[new]);
          }
  #endif

          // update correct solution

          sigma = sqrt(2 . * kappa * time + sigmao*sigmao);
          a = ao*sigmao/sigma;
          for (i=0; i<totpoints+2; i++) {
              theory[i] = a*exp(-(x[i] * x[i]) / (2. * sigma*sigma));
          }

  #ifdef PGPLOT
          if (step % plotsteps == 0) {
              cpgsls(1); cpgslw(6); cpgsci(white);
              cpgline(totpoints+2, x, theory);
              cpgebuf();
          }
  #endif
          error = 0.;
          for (i=1;i<totpoints+1;i++) {
              error += (theory[i] - temperature[new][i]) * (theory[i] - temperature[new][i]);
          }
          error = sqrt(error);

              printf("Step = %d, Time = %g, Error = %g\n", step, time, error);

          old = new;
          new = 1 - old;
      }

      // free data

      free(temperature[1]);
      free(temperature[0]);
      free(temperature);
      free(x);
      free(theory);

      return 0;
  }
```
### Modify the code to work with MPI
In the domain decomposition, the stencils will jut out into a neighbouring subdomain. If we fill the guard cells with values from the neighbouring stencils, then we treat each coupled subdomain as independent with boundary conditions. Therefore, we need a communicator.

For now, we modify the above program in a few ways:
 * Decompose the global domain into a smaller piece for each process. Each process needs to know where it is in the domain to aply initial conditions.
 * Pass the guard cell values to appropriate neighbours using `MPI_Sendrecv`.

We add these into the code with the following snippets (of course also including declarations of all variables)

Include the usual first MPI calls

```C
ierr = MPI_Init(&argc, &argv);
ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
ierr = MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
```

Compute the number of points in each subdomain, and use `locpoints` to allocate memory.
```C
locpoints = totpoints/size;
```

Find the neighbours, and set to `MPI_PROC_NULL` if it is an edge
```C
left = rank-1;
if (left < 0) left = MPI_PROC_NULL;
right= rank+1;
if (right >= size) right = MPI_PROC_NULL;
```

The boundary conditions are the tricky part. We use send and receive the guard cell values before computing all the new values.

```C
for (step=0; step < nsteps; step++) {
    // boundary conditions: keep endpoint temperatures fixed.

    temperature[old][0] = fixedlefttemp;
    temperature[old][locpoints+1] = fixedrighttemp;

    // send data rightwards
    ierr = MPI_Sendrecv(&(temperature[old][locpoints]), 1, MPI_FLOAT, right, righttag,
                 &(temperature[old][0]), 1, MPI_FLOAT, left,  righttag, MPI_COMM_WORLD, &status);

    // send data leftwards
    ierr = MPI_Sendrecv(&(temperature[old][1]), 1, MPI_FLOAT, left, lefttag,
                 &(temperature[old][locpoints+1]), 1, MPI_FLOAT, right,  lefttag, MPI_COMM_WORLD, &status);


    for (i=1; i<locpoints+1; i++) {
        temperature[new][i] = temperature[old][i] + dt*kappa/(dx*dx) *
            (temperature[old][i+1] - 2. * temperature[old][i] +
             temperature[old][i-1]) ;
    }


    time += dt;
  }
```

## Non-blocking communication
Consider the above example. The processors had to wait for the send/receive before doing their useful computation. The sequence of communication and computation was

 * Code exchanges guard cells using `Sendrecv`
 * The code **then** computes the next step
 * Then again exchanges guard cells
 * And repeat

This isn't the only computation/communication pattern we can use. Instead, we could use a pattern such as

 * Start a send of guard cells
 * Without waiting for that send's completion, the code computes the next step for the inner cells, while the guard cell message is in transit
 * The code receives the guard cells
 * Afterwards, it computes the other cell's new values
 * Repeat

This pattern is called **non-blocking communication**. The communication does not block the useful computations. As usual, MPI implements functions for this for us. These functions are `MPI_Isend` and `MPI_Irecv`. These functions initiate the communication, but do not block the computation from proceeding. Some details about the functions:

 * `MPI_Isend(sendptr, count, MPI_TYPE, destination, tag, Communicator, MPI_Request)`
   * `sendptr`/`recvptr`: pointer to message
   * `count`: number of elements in the ptr
   * `MPI_TYPE`: MPI datatype
   * `destination`/`source`: rank of sender/receiver
   * `tag`: unique ID for message pair
   * `Communicator`: `MPI_COMM_WORLD` or user created
   * `MPI_Request`: Identify comm operations (`MPI_Request` type)
 * `MPI_Irecv`
   * See above for definitions

Of course if the program continues without knowing if the sends/receives are finished, at some point this can cause a problem. If we try to use the guard cell values before we have finished receiving them, the program will crash. MPI has various **wait** functions for this. These functions block the process from continuing until the request(s) they are watching are finished. Two of the wait functions are
 * `MPI_Wait(MPI_Request, MPI_Status)`
 * `MPI_Waitall(count, MPI_Request, MPI_Status)`

Where
  * `MPI_Request` identifies the comm operation(s)
  * `MPI_Status` is the status of comm operation(s)
  * `flag`: `true` if the comm is complete, `false` if not sent/received yet

We can further improve the above  example by using the `Isend`/`Irecv` and wait functions. We need to wait for the  sends/receives to finish before we can use the guard cells in computations, but we can compute the middle of the domains perfectly fine while the communication is in transit. The new part of the code is

```C
MPI_Request request[4];
MPI_Status status[4];
for (step=0; step < nsteps; step++) {
    temperature[old][0] = fixedlefttemp;
    temperature[old][locpoints+1] = fixedrighttemp;

  ierr = MPI_Isend(&(temperature[old][locpoints]), 1, MPI_FLOAT, right, righttag, MPI_COMM_WORLD, &request[0]);
  ierr = MPI_Isend(&(temperature[old][1]), 1, MPI_FLOAT, left, lefttag, MPI_COMM_WORLD, &request[1]);
  ierr = MPI_Irecv(&(temperature[old][0]), 1, MPI_FLOAT, left, righttag, MPI_COMM_WORLD, &request[2]);
  ierr = MPI_Irecv(&(temperature[old][locpoints+1]), 1, MPI_FLOAT, right, lefttag, MPI_COMM_WORLD, &request[3]);

  for (i=2; i<locpoints; i++) {
      temperature[new][i] = temperature[old][i] + dt*kappa/(dx*dx) *
          (temperature[old][i+1] - 2. * temperature[old][i] +
           temperature[old][i-1]) ;
      }

  ierr = MPI_Waitall(4, &request, &status); // important to wait here!

  i = 1;
      temperature[new][i] = temperature[old][i] + dt*kappa/(dx*dx) *
      (temperature[old][i+1] - 2. * temperature[old][i] + temperature[old][i-1]) ;

  i = locpoints+1;
  temperature[new][i] = temperature[old][i] + dt*kappa/(dx*dx) *
  (temperature[old][i+1] - 2. * temperature[old][i] + temperature[old][i-1]) ;

  time += dt;
}

```

## MPI-IO
File I/O is the most expensive part of our workflow. We would like I/O to be parallel not serial, but writing one file per process is inconvenient and inefficient. However, having multiple processes write to the same file is difficult and makes files corrupt.

As usual, the MPI standard has a solution for this: MPI-IO. Packages like netCDF and HDF5 are built on top of this standard, and so they can write in parallel without having to deal with all the low-level details.

MPI-IO exploits analogies with MPI:

 * Writing ~ sending message
 * Reading ~ receiving message
 * File access grouped via communicator: collective operations
 * User defined MPI datatypes
 * IO latency hiding much like communicator latency hiding
 * All functionality through function calls

As in regular I/O, files are maintained through file handles. A file gets opened with `MPI_File_open`, eg.

```C
MPI_File fh;
MPI_File_open(MPI_COMM_WORLD, "test.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
MPI_File_close(&fh);
```

The main open modes are:

 * `MPI_MODE_RDONLY`: read only
 * `MPI_MODE_RDWR`: reading and writing
 * `MPI_MODE_CREATE`: create the file if it does not exist

More exist, but they are more specialized to certain use cases.

To make binary access more natural, MPI-IO defines file access through the **file view**. The file view is specified by

 * displacement: Where to start in the file
 * etype: Allows to access the file in units other than bytes
 * filetype: Each process defines what part of a shared file it uses.

### Example of file IO

```C
#include <string.h>
#include <mpi.h>

int rank, size, msgsize;
char message;

MPI_File file;
MPI_Status status;
MPI_Offset offset;

msgsize = 6;

MPI_Init();
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_world(MPI_COMM_WORLD, &rank);

if (rank % 2) strcpy(message, "World!"); else strcpy(message, "Hello ");
offset = msgsize * rank;
MPI_File_open(MPI_COMM_WORLD, "helloworld.txt", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
MPI_File_seek(file, offset, MPI_SEEK_SET);
MPI_File_write(file, message, msgsize, MPI_CHAR, &status);
MPI_File_close(&file);
MPI_Finalize();
```
