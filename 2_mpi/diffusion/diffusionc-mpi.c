#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cpgplot.h"
#include <mpi.h>

int main(int argc, char **argv) {
    /* simulation parameters */
    const int totpoints=12;
    const float xleft = -12., xright = +12.;
    const float kappa = 1.;

    const int nsteps=1;
    const int plotsteps=50;

    /* data structures */
    float *x;
    float **temperature;
    float *theory;

    /* parameters of the original temperature distribution */
    const float ao=1., sigmao=1.;
    float a, sigma;

    float fixedlefttemp, fixedrighttemp;

    int old, new;
    int step, i;
    int red, grey, white;
    float time;
    float dt, dx;
    float error;

    // MPI definitions
    float ierr;
    int numtasks, rank;

    int numpoints;

    int lefttag, righttag;


    // initialize MPI communicator

    ierr = MPI_Init(&argc, &argv);

    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ierr = MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    printf("Rank: %i of %i \n", rank, numtasks);
    dx = (xright-xleft)/(totpoints);
    dt = dx*dx * kappa/10.;
    numpoints = totpoints / numtasks;
    printf("Using %i points \n", numpoints);

    /*
     * allocate data, including ghost cells: old and new timestep
     * theory doesn't need ghost cells, but we include it for simplicity
     */

    theory = (float *)malloc((numpoints+2)*sizeof(float));
    x      = (float *)malloc((numpoints+2)*sizeof(float));
    temperature = (float **)malloc(2 * sizeof(float *));
    temperature[0] = (float *)malloc((numpoints+2)*sizeof(float));
    temperature[1] = (float *)malloc((numpoints+2)*sizeof(float));
    old = 0;
    new = 1;

    left = rank - 1;
    if (left < 0) left = MPI_PROC_NULL;

    right = right + 1;
    if (right > numtasks) right = MPI_PROC_NULL;

    time = 0.0;
    for (i=0;i<numpoints+2;i++) {
	x[i] = xleft + (rank*numpoints + i -1 + 0.5)*dx;
        printf("(%i) %f ", rank, x[i]);
	temperature[old][i] = ao*exp(-(x[i]*x[i]) / (2.*sigmao*sigmao));
        theory[i]           = ao*exp(-(x[i]*x[i]) / (2.*sigmao*sigmao));
    }

    lefttemp = ao*exp(-(xleft-dx)*(xleft-dx) / (2.*sigmao*sigmao));
    righttemp= ao*exp(-(xright+dx)*(xright+dx)/(2.*sigmao*sigmao));

    printf("\n\n");
    
    // time-evolution
    
    for (step=0; step<nsteps; step++) {
	temperature[old][0] =           lefttemp;
        temperature[old][numpoints+1] = righttemp;

        ierr = MPI_Sendrecv(&(temperature[old][numpoints]), 1, MPI_FLOAT, right, righttag, &(temperature
[old][0]), 1, MPI_FLOAT, left, righttag, MPI_COMM_WORLD, &status);

        ierr = MPI_sendrecv(&(temperature[old][1]), 1, MPI_FLOAT, left, lefttag, &(temperature[old][nump
oints+1]), 1, MPI_FLOAT, right, lefttag, MPI_COMM_WORLD, &status);

	for (i=1; i<totpoints+1; i++) {
	    temperature[new][i] = temperature[old][i-1] + dt*kappa/(dx*dx) * 
		(temperature[old][i+1] - 2*temperature[old][i] + temperature[old][i+1]);
		}
	printf("Step: %i", step);

	time += dt
    return 0;
}
