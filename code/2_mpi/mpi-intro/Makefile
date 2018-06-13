#all: minmeanmax minmeanmax-allreduce minmeanmax-mpi
include ../Makefile.inc

messages: firstmessagec secondmessagec thirdmessagec fourthmessagec fifthmessagec firstmessagef secondmessagef thirdmessagef fourthmessagef fifthmessagef

hello-worldc: hello-world.c	
	${MPICC} ${CFLAGS} -o $@ $<

hello-worldf: hello-world.f90
	${MPIF90} ${FFLAGS} -g -o $@ $<

firstmessagec: firstmessage.c
	${MPICC} ${CFLAGS} -o $@ $<

secondmessagec: secondmessage.c
	${MPICC} ${CFLAGS} -o $@ $<

thirdmessagec: thirdmessage.c
	${MPICC} ${CFLAGS} -o $@ $<

fourthmessagec: fourthmessage.c
	${MPICC} ${CFLAGS} -o $@ $<

fifthmessagec: fifthmessage.c
	${MPICC} ${CFLAGS} -o $@ $<

firstmessagef: firstmessage.f90
	${MPIF90} ${FFLAGS} -o $@ $<

secondmessagef: secondmessage.f90
	${MPIF90} ${FFLAGS} -o $@ $<

thirdmessagef: thirdmessage.f90
	${MPIF90} ${FFLAGS} -o $@ $<

fourthmessagef: fourthmessage.f90
	${MPIF90} ${FFLAGS} -o $@ $<

fifthmessagef: fifthmessage.f90
	${MPIF90} ${FFLAGS} -o $@ $<

minmeanmax: minmeanmax.f90
	${F90} ${FFLAGS} -o $@ $<

minmeanmax-mpi: minmeanmax-mpi.f90
	${MPIF90} ${FFLAGS} -g -o $@ $<

minmeanmax-allreduce: minmeanmax-allreduce.f90
	${MPIF90} ${FFLAGS} -o $@ $<


clean:
	rm -rf minmeanmax
	rm -rf minmeanmax-mpi
	rm -rf minmeanmax-allreduce
	rm -rf firstmessagec secondmessagec thirdmessagec fourthmessagec fifthmessagec
	rm -rf *.o
	rm -rf *~
