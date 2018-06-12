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

      printf("rank= %d  Results: %f %f %f %f\n",rank,recvbuf[0],
             recvbuf[1],recvbuf[2],recvbuf[3]);

      MPI_Gather(recvbuf, SIZE, MPI_FLOAT, newbuf, SIZE,
                  MPI_FLOAT, source, MPI_COMM_WORLD)

      if ( rank == source) {
        for (int i = 0;i<SIZE; i++)
          for (int j=0;j<SIZE; j++)
            printf("%f ", newbuf[i][j]);
          printf("\n")
        }
      }
  else
  printf("Must specify %d processors. Terminating.\n",SIZE);
  MPI_Finalize();
  }
