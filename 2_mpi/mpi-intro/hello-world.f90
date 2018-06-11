program helloworld
use mpi
implicit none
integer :: rank, comsize, ierr

call MPI_Init(ierr)
call MPI_Comm_size(MPI_COMM_WORLD, comsize, ierr)
call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
print *,"Hello, world from task ",rank, &
        ' of ', comsize

call MPI_Finalize(ierr)
end program helloworld


