/* CUDA reduction exercise (summation in this case). The initial code
   reduction.cu is a CUDA code using the most primitive type of
   reduction - purely atomic.

   Your task is to

   1) Compile both the serial code and the purely atomic code, and
   measure the (in)efficiency of the purely atomic solution.

   2) Modify reduction.cu to implement a hybrid reduction approach:
   binary reduction at the low level (beginning of the kernel), atomic
   reduction at the top level (end of the kernel). More specifically, the
   results of binary reductions should be added up globally using
   atomicAdd function at the end of the kernel.


   The code always computes the "exact result" sum0 (using double
   precision, serially) - don't touch this part, it is needed to
   estmate the accuracy of your computation.

   The initial copying of the array to device is not timed. We are
   only interested in timing different reduction approaches.

   At the end, you will have to copy the reduction result (sum) from
   device to host, using cudaMemcpyFromSymbol.

   You will discover that for large NMAX, atomic summation is much
   slower than serial code. How about hybrid reduction?


To compile on monk:

  nvcc -arch=sm_20 -O2 reduction.cu -o reduction

To compile on graham:
  module load cuda
  nvcc -arch=sm_60 -O2 reduction.cu -o reduction

The speedup on graham (vs reduction_serial) is ~26x.
  
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
#define NMAX 2097152

#define NBLOCKS NMAX/BLOCK_SIZE

// Input array (global host memory):
float h_A[NMAX];
// Copy of h_A on device:
__device__ float d_A[NMAX];

__device__ float d_sum;


/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  */

// It messes up with y!

int
timeval_subtract (double *result, struct timeval *x, struct timeval *y)
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


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// Kernel(s) should go here:


// The only purpose of the kernel is to initialize one global variable, d_sum
__global__ void init_kernel ()
{
  d_sum = 0.0;
  return;
}


__global__ void MyKernel ()
{

  __shared__ float sum[BLOCK_SIZE];

  int i = threadIdx.x + blockDim.x * blockIdx.x;

//  sum[i] = d_A[i];
  sum[threadIdx.x] = d_A[i];
  // Not needed, because NMAX is a power of two:
  //  if (i >= NMAX)
  //    return;
  
  __syncthreads();

  int nTotalThreads = blockDim.x;

  while(nTotalThreads > 1) {
    int halfPoint = nTotalThreads / 2;

    if (threadIdx.x < halfPoint) {
      int thread2 = threadIdx.x + halfPoint;
      sum[threadIdx.x] += sum[thread2];
    }
    __syncthreads();
    nTotalThreads = halfPoint;
  }

  if (threadIdx.x == 0) {
    atomicAdd(&d_sum, sum[0]);
  }
  
  return;
}




int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double sum0, restime;
  float sum;
  int devid, devcount, error;

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
      printf ("Device count, devid: %d %d\n", devcount, devid);
      printf ("Device: %s\n", deviceProp.name);
      printf("[deviceProp.major.deviceProp.minor] = [%d.%d]\n\n", deviceProp.major, deviceProp.minor);
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
  if (error = cudaMemcpyToSymbol( d_A, h_A, NMAX*sizeof(float), 0, cudaMemcpyHostToDevice))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

// Set d_A to zero on device:
  init_kernel <<< 1,1 >>> ();

  //--------------------------------------------------------------------------------
  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr0, NULL);


  // The kernel call:
  MyKernel <<<NBLOCKS, BLOCK_SIZE>>> ();


  // Copying the result back to host (we time it):
  if (error = cudaMemcpyFromSymbol (&sum, d_sum, sizeof(float), 0, cudaMemcpyDeviceToHost))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);
  if (kk == 0)
    printf ("Sum: %e (relative error %e)\n", sum, fabs((double)sum-sum0)/sum0);

  avr = avr + restime;
  //--------------------------------------------------------------------------------

} // kk loop
  printf ("\n Average time: %e\n", avr/NTESTS);

  return 0;

}
