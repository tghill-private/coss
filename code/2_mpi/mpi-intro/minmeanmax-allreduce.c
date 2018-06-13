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
