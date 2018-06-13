# Programming GPUs with CUDA - Day 1
*June 13, 2018. University of Toronto, Wilson Hall. Pawel Pomorski*

## Introduction to GPUs
GPUs started out being used entirely for enabling graphics for computer games. However, now they are responsible for much of the computing power of the most powerful supercomputers (eg. Summit).

NVIDIA offers **CUDA** for programming GPUs, while AMD uses **OpenCL**. CUDA is more advanced, but it is controlled by NVIDIA. OpenCL is an open source alternative, but it lags behind CUDA.

### What is the difference between CPUs and GPUs?
CPUs are general purpose, meant for task parallelism, and are meant to minimize latency. Much of the CPU is taken up by the control system and cache, not just for the actual computation.

GPUs excel at number crunching, are meant for data parallelism, and to maximize throughput. Most of the GPU is devoted to the actual computation. The control system is much more basic, so we have to be careful how we use the available cores.

The ideal usage pattern for GPUs is stream computing: use a kernal, eg. y_i = x_i + 1, and apply this to each thread from an input stream.

## CUDA programming model
The main CPU is called the *host*. The compute device (GPU) is viewed as a coprocessor capable of executing a large number of lightweight threads in parallel. Computation on the device is performed by *kernels*, functions executed in parallel on each data element. The host manages all the memory allocations and invocations of kernels on the device.

The basic paradigm is
 * host uploads inputs to device
 * host remains busy while device performs computation
 * host downloads results

## Introduction to CUDA
CUDA is essentially an extension of the C programing language. CUDA source code is saved in `.cu` files. Source code is compiled to object files using `nvcc` compiler.

### Example 1
SAXPY (Scalar Alpha X Plus Y) is the linear algebra operation

```
y = a * x + y
```

The basic C implementation would be

```C
void saxpy_cpu(float * vecY, float * vecX, float alpha, int n) {
  int i;

  for (i=0; i<n; i++) {
    vecY[i] = alpha * vecX[i] + vecY[i];
  }
}
```

The CUDA kernel function implementing SAXPY is

```C
__global__ void saxpy_gpu(float * vecY, float * vecX, float alpha, int n) {
  int i;

  i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i<n) vecY[i] = alpha * vecX[i] + vecY[i];
}
```

**Remarks**

 * The `__global__` qualifier identifies this function as a kernel that executes on the device
 * `blockIdx`, `blockDim`, and `threadIdx` are built-in variables that uniquely identify a thread's position in the execution environment. They are used to compute an offset into the data array.
 The structure in 1-D is
 ```
 |  Block 0  | Block 1    | Block 2   | Block 3   | ...   |
 | --------- | ---------- | --------- | --------- | ---   |
```
  Where `blockIdx` says which block we are in (0, 1, 2, 3, ...), `blockDim` says how big each block is, and `threadIdx` says where we are in the block.

 * The host specifies the number of blocks and block size during kernel invocation
  ```C
  saxpy_gpu<<numBlocks, blockSize>>(y_d, x_d, alpha, n)
  ```

Consider the implementation with cuda below:
```C
  /*
Implementation of SAXPY accelerated with CUDA.

A CPU implementation is also included for comparison.

No timing calls or error checks in this version, for clarity.

Compile on graham with:

nvcc -arch=sm_60 -O2 saxpy_cuda.cu

nvprof ./a.out


*/


#include <cuda.h> /* CUDA runtime API */
#include <cstdio>
void saxpy_cpu(float *vecY, float *vecX, float alpha, int n) {
    int i;

    for (i = 0; i < n; i++)
        vecY[i] = alpha * vecX[i] + vecY[i];
}

__global__ void saxpy_gpu(float *vecY, float *vecX, float alpha ,int n) {
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
        vecY[i] = alpha * vecX[i] + vecY[i];
}


int main(int argc, char *argv[]) {
    float *x_host, *y_host;   /* arrays for computation on host*/
    float *x_dev, *y_dev;     /* arrays for computation on device */
    float *y_shadow;          /* host-side copy of device results */

    int n = 1024*1024;
    float alpha = 0.5f;
    int nerror;

    size_t memsize;
    int i, blockSize, nBlocks;

    memsize = n * sizeof(float);

    /* allocate arrays on host */

    x_host = (float *)malloc(memsize);
    y_host = (float *)malloc(memsize);
    y_shadow = (float *)malloc(memsize);

    /* allocate arrays on device */

    cudaMalloc((void **) &x_dev, memsize);
    cudaMalloc((void **) &y_dev, memsize);

    /* catch any errors */

    /* initialize arrays on host */

    for ( i = 0; i < n; i++) {
        x_host[i] = rand() / (float)RAND_MAX;
        y_host[i] = rand() / (float)RAND_MAX;
    }

    /* copy arrays to device memory (synchronous) */

    cudaMemcpy(x_dev, x_host, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(y_dev, y_host, memsize, cudaMemcpyHostToDevice);

    /* set up device execution configuration */
    blockSize = 512;
    nBlocks = n / blockSize + (n % blockSize > 0);

    /* execute kernel (asynchronous!) */

    saxpy_gpu<<<nBlocks, blockSize>>>(y_dev, x_dev, alpha, n);

    /* execute host version (i.e. baseline reference results) */
    saxpy_cpu(y_host, x_host, alpha, n);

    /* retrieve results from device (synchronous) */
    cudaMemcpy(y_shadow, y_dev, memsize, cudaMemcpyDeviceToHost);

    /* guarantee synchronization */
    cudaDeviceSynchronize();

    /* check results */
    nerror=0;
    for(i=0; i < n; i++) {
        if(y_shadow[i]!=y_host[i]) nerror=nerror+1;
    }
    printf("test comparison shows %d errors\n",nerror);

    /* free memory */
    cudaFree(x_dev);
    cudaFree(y_dev);
    free(x_host);
    free(y_host);
    free(y_shadow);

    return 0;
}
```

** Remarks**

 * This code works, but it is not an ideal usage of the GPU device. The data transfer between host and device is very slow compared to the host or device accessing their own RAM. For a computation like we have done, it takes longer to copy the data over than to do the computation.
 * We should include error checking, but this code ignores it for sake of clarity. We should have something like
 ```C
 ...
/* check CUDA API function call for possible error */
if (error = cudaMemcpy(x_dev, x_host, memsize, cudaMemcpyHostToDevice))
{
printf ("Error %d\n", error);
exit (error);
}
...
saxpy_gpu<<<nBlocks, blockSize>>>(y_dev, x_dev, alpha, n);
/* make sure kernel has completed*/
cudaDeviceSynchronize();
/* check for any error generated by kernel call*/
if(error = cudaGetLastError())
{
printf ("Error detected after kernel %d\n", error);
exit (error);
}
...
```

### Timing a GPU program
It is of course useful to be able to time a GPU program, since the whole point of writing CUDA code is to speed up your program. The following code snippet can be useful

```C
...
cudaEvent_t start, stop;
float kernel_timer;
...
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);
saxpy_gpu<<<nBlocks, blockSize>>>(y_dev, x_dev, alpha, n);
cudaEventRecord(stop, 0);
cudaEventSynchronize( stop );
cudaEventElapsedTime( &kernel_timer, start, stop );
printf("Test Kernel took %f ms\n",kernel_timer);
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

But there are some things to keep in mind when timing your code:

 * There is overhead to "spinning up" a GPU. The first call may be very slow, so it's a good strategy to have a "warmup" run. And using an average is essential.

#### Memory allocation
There is a concept called pinned memory in CUDA. This memory on the host can transfer to the device faster. The CUDA function forthis is `cudaMallocHost((void **) &a_host, memsize)`, and the memory must be freed using `cudaFree(a_host)`. This is an easy way to slightly speed up your code.


## CUDA Extension to C
Beyond `__global__` there are some other qualifiers:
 * `__device__`: Device functions only callable from device
 * `__host__`  : Host functions only callable from host (default for no qualifier)
 * `__shared__`: Memory shared bya block of threads executing on a mulitprocessor
 * `__constant__`: Special memory for constants (cached)


 ### Kernel Launching
 Launch kernels with the syntax

 ```C
 myKernel<<GridDef, BlockDef>>(paramlist);
 ```
 Where `GridDef` and `BlockDef` can be specified as `dim3` objects. Grids and blocks can be 1D, 2D, or 3D.

 **Example**: 2D addressing, 10x10 blocks with 16x16 threads per block

```C
dim3 gridDef(10, 10, 1);
dim2 blockDef(16, 16, 1);
kernel<<gridDef, blockDef>>(paramlist);
```

## Example: Julia set
Modify the CPU program `julia_cpu.cu` to run on GPUs with CUDA.

Serial program:
```C
/*
Code adapted from book "CUDA by Example: An Introduction to General-Purpose GPU Programming"

This code computes a visualization of the Julia set.  Two-dimensional "bitmap" data which can be plotted is computed by the function kernel.

The data can be viewed with gnuplot.

The Julia set iteration is:

z= z**2 + C

If it converges, then the initial point z is in the Julia set.

This code is CPU only but will compile with:

nvcc julia_cpu.cu


*/


#include <stdio.h>

#define DIM 1000

int julia( int x, int y ) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    float cr=-0.8f;
    float ci=0.156f;

    float ar=jx;
    float ai=jy;
    float artmp;

    int i = 0;
    for (i=0; i<200; i++) {

        artmp = ar;
        ar =(ar*ar-ai*ai) +cr;
        ai = 2.0f*artmp*ai + ci;

        if ( (ar*ar+ai*ai) > 1000)
            return 0;
    }

    return 1;
}

void kernel( int *arr ){
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            arr[offset] = juliaValue;
        }
    }
}

int main( void ) {
    int arr[DIM*DIM];
    FILE *out;
    kernel(arr);

    out = fopen( "julia.dat", "w" );
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;
            if(arr[offset]==1)
                fprintf(out,"%d %d \n",x,y);
        }
    }
    fclose(out);

}
```
** We can not do much to parallelize the `int julia` function. However, we can parallelize the `kernel` function.

Based on the GPU architecture, we have a maximum block size `blocksize = 32`. We will add this as a constant into the `main` function.

We CUDA-ize the code by:

* define `julia` function as a `__device__` function
   ```C
   __device__ int julia( int x, int y )
   ...
  ```

* Turn `kernel` function into a true Kernel
```C
__global__ void kernel(int *arr){

    int x, y, i;

    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < DIM && y < DIM) {
      i = x + y * DIM;
      int juliaValue = julia( x, y );
      arr[i] = juliaValue;
    }
}
```
* Add memory allocations and kernel call to `main` function.

```C
int main( void ) {
    int arr_host[DIM*DIM];
    int * arr_dev;
    int blocksize = 32;
    FILE * out;


    dim3 gridDef(DIM/blocksize + 1, DIM/blocksize + 1, 1);
    dim3 blockDef(blocksize, blocksize, 1);

    size_t memsize;

    memsize = DIM*DIM*sizeof(int);

    cudaMallocHost((void ** ) &arr_host, memsize);
    cudaMalloc((void ** ) &arr_dev, memsize);

    cudaMemcpy(arr_dev, arr_host, memsize, cudaMemcpyHostToDevice);

    kernel<<<gridDef, blockDef>>>(arr_dev);

    cudaMemcpy(arr_host, arr_dev, memsize, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    out = fopen( "julia.dat", "w" );
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;
            if(arr_host[offset]==1)
                fprintf(out,"%d %d \n",x,y);
        }
    }
    fclose(out);

    cudaFree(arr_host);
    cudaFree(arr_dev);
}
```

We should really include error checking in this program as well; see above code snippets for how to include that.

This program is not fully optimized for GPU computing. When we invoke the kernel, some of the threads will execute the maximum number of iterations, while some will execute less. In other words, the threads are not load balanced. Moreover, this program is fairly light. Most of the time is spent uploading and downloading data, not in the actual computing.
