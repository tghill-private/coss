       program randomdata
       implicit none
       integer,parameter :: nx=1500
       real, allocatable :: dat(:)

       integer :: i
       real    :: datamin, datamax, datamean

!
! random data
!
       allocate(dat(nx))
       call srand(0)
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
       deallocate(dat)

       print *,'min/mean/max = ', datamin, datamean, datamax
  
       return
       end


  
