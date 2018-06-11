       program randomdata
       use mpi
       implicit none
       integer,parameter :: nx=1500
       real, allocatable :: dat(:)

       integer :: i
       real    :: datamin, datamax, datamean
       integer :: ierr, rank, comsize
       integer :: ourtag=5
       real :: sendbuffer(3)
       real :: recvbuffer(3)
       integer :: status

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

       if (rank /= 0) then
            sendbuffer(1) = datamin
            sendbuffer(2) = datamean
            sendbuffer(3) = datamax
            call MPI_SSEND(sendbuffer, 3, MPI_REAL, 0, ourtag, MPI_COMM_WORLD, ierr)
       else
            do i=1,comsize-1 
                call MPI_RECV(recvbuffer, 3, MPI_REAL, MPI_ANY_SOURCE, ourtag, MPI_COMM_WORLD, status, ierr)
                if (recvbuffer(1) < datamin) datamin=recvbuffer(1)
                if (recvbuffer(3) > datamax) datamax=recvbuffer(3)
                datamean = datamean + recvbuffer(2)
            enddo
            datamean = datamean / comsize
       endif
       deallocate(dat)

       print *,'min/mean/max = ', datamin, datamean, datamax

       if (rank==0) then
           print *, 'global min/mean/max = ', datamin, datamean, datamax
       endif 
  
       call MPI_FINALIZE(ierr)
       end


  
