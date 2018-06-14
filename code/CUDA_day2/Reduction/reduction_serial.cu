/* CUDA reduction exercise. The serial version (written as *.cu
file). Use it to compare your CUDA code performance to a serial code
performance.

To compile on monk:

  nvcc -arch=sm_20 -O2 reduction.cu -o reduction

To compile on graham:
  module load cuda
  nvcc -arch=sm_60 -O2 reduction.cu -o reduction

*/

#include <sys/time.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Number of times to run the test (for better timings accuracy):
#define NTESTS 100

// Number of threads in one block (possible range is 32...1024):
#define BLOCK_SIZE 256

// Total number of threads (total number of elements to process in the kernel):
// For simplicity, use a power of two:
#define NMAX 2097152

// Number of blocks
// This will be needed for the second kernel in two-step binary reduction
// (to declare a shared memory array)
#define NBLOCKS NMAX/BLOCK_SIZE


// Input array (global host memory):
float h_A[NMAX];
// Copy of h_A on device:
__device__ float d_A[NMAX];

int timeval_subtract (double *result, struct timeval *x, struct timeval *y);


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// Kernel(s) should go here:


int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double sum0, restime;
  float sum;
  int devid, devcount, error;

  if (BLOCK_SIZE>1024)
    {
      printf ("Bad BLOCK_SIZE: %d\n", BLOCK_SIZE);
      exit (1);
    }

  /* find number of device in current "context" */
  cudaGetDevice(&devid);
  /* find how many devices are available */
  if (cudaGetDeviceCount(&devcount) || devcount==0)
    {
      printf ("No CUDA devices!\n");
      exit (1);
    }
  else
    {
      cudaDeviceProp deviceProp; 
      cudaGetDeviceProperties (&deviceProp, devid);
    }

// Loop to run the timing test multiple times:
  double avr = 0.0;
  int kk;
for (kk=0; kk<NTESTS; kk++)
{

  // We don't initialize randoms, because we want to compare different strategies:
  // Initializing random number generator:
  //  srand((unsigned)time(0));

  // Initializing the input array:
  for (int i=0; i<NMAX; i++)
    {
      h_A[i] = (float)rand()/(float)RAND_MAX;
    }

  // Don't modify this: we need the double precision result to judge the accuracy:
  sum0 = 0.0;
  for (int i=0; i<NMAX; i++)
    sum0 = sum0 + (double)h_A[i];

  // Copying the data to device (we don't time it):
  if (error = cudaMemcpyToSymbol (d_A, h_A, NMAX*sizeof(float), 0, cudaMemcpyHostToDevice))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  //--------------------------------------------------------------------------------
  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr0, NULL);


  // This serial summation will have to be replaced by calls to kernel(s):
  sum = 0.0;
  for (int i=0; i<NMAX; i++)
    sum = sum + h_A[i];


  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);

  // We are printing the result here, after cudaDeviceSynchronize (this will matter
  // for CUDA code - why?)
  if (kk == 0)
    printf ("Sum: %e (relative error %e)\n", sum, fabs((double)sum-sum0)/sum0);

  avr = avr + restime;
  //--------------------------------------------------------------------------------

} // kk loop
  printf ("\n Average time: %e\n", avr/NTESTS);

  return 0;

}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  */

// It messes up with y!

int timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
  struct timeval result0;

  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

