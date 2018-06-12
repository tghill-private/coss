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
### Guard cell exchange
In the domain decomposition, the stencils will jut out into a neighbouring subdomain. If we fill the guard cells with values from the neighbouring stencils, then we treat each coupled subdomain as independent with boundary conditions. Therefore, we need a communicator.

### Hands-on: Modify the code to work with MPI
The hard part of making this code work is setting the boundary conditions. (TODO)

## Non-blocking communication
These are a mechanism for overlapping/interleaving communications and useful computations. For instance, in the diffusion example the processors had to wait for the send/receive before doing their useful computation. We want to be able to simultaneously carry our useful computation and be performing sends/receives.

In particular, the sequence of communicaiton and computation was

 * Code exchanges guard cells using `Sendrecv`
 * The code **then** computes the next step
 * Then again exchanges guard cells
 * And repeat

A non-blocking communication/computation pattern is

 * Start a send of guard cells using `ISend`
 * Without waiting for that send's completion, the code computes the next step for the inner cells, while the guard cell message is in transit
 * The code receives the guard cells using `IRecv`
 * Afterwards, it computes the other cell's new values
 * Repeat

The functions that implement this are

 * `MPI_Isend(sendptr, count, MPI_TYPE, destination, tag, Communicator, MPI_Request`
   * `sendptr`/`recvptr`: pointer to message
   * `count`: number of elements in the ptr
   * `MPI_TYPE`: MPI datatype
   * `destination`/`source`: rank of sender/receiver
   * `tag`: unique ID for message pair
   * `Communicator`: `MPI_COMM_WORLD` or user created
   * `MPI_Request`: Identify comm operations
 * `MPI_Irecv`
   * See above for definitions

We can tell if the message is completed by the wait functions
 * `MPI_Wait(MPI_Request, MPI_Status)`
 * `MPI_Waitall(count, MPI_Request, MPI_Status)`
Where
 * `MPI_Request` identifies the comm operation(s)
 * `MPI_Status` is the status of comm operation(s)
 * `flag`: `true` if the comm is complete, `false` if not sent/received yet
