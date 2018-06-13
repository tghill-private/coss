program randomdata
use mpi
implicit none
integer,parameter :: nx=1500
real, allocatable :: dat(:)

integer :: i
real    :: datamin, datamax, datamean
real    :: globalmin, globalmean, globalmax
integer :: ierr, rank, comsize
integer :: ourtag=5
integer,dimension(MPI_STATUS_SIZE) :: rstatus

call MPI_INIT(ierr)
call MPI_COMM_SIZE(MPI_COMM_WORLD, comsize, ierr)
call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
!
! random data
!
allocate(dat(nx))
call srand(rank)
do i=1,nx
  dat(i) = 2*rand(0)-1.
enddo

!
! find min/mean/max
! 
datamin = 1e+19
datamax =-1e+19
datamean = 0.

do i=1,nx
  if (dat(i) .lt. datamin) datamin = dat(i)
  if (dat(i) .ge. datamax) datamax = dat(i)
  datamean = datamean + dat(i)
enddo
datamean = datamean/(1.*nx)

call MPI_Allreduce(datamin, globalmin, 1, MPI_REAL, &
                   MPI_MIN, MPI_COMM_WORLD, ierr)
!
! to just sent to rank 0:
! call MPI_Reduce(datamin, globalmin, 1, MPI_REAL, &
!                 MPI_MIN, 0, MPI_COMM_WORLD, ierr)
!
call MPI_Allreduce(datamax, globalmax, 1, MPI_REAL, &
                   MPI_MAX, MPI_COMM_WORLD, ierr)
call MPI_Allreduce(datamean, globalmean, 1, MPI_REAL, &
                   MPI_SUM, MPI_COMM_WORLD, ierr)

globalmean = globalmean/comsize

print *,'min/mean/max = ', datamin, datamean, datamax

if (rank==0) then
   print *, 'global min/mean/max = ', globalmin, globalmean, globalmax
endif 

call MPI_FINALIZE(ierr)
end



