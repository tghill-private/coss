include ../Makefile.inc

all: sine 

sine: sine.c
	${MPICC} ${CFLAGS} -lm -o $@ $<

sinef: sine.f90
	${MPIF90} ${FFLAGS} -g -o $@ $<

sineplot: sine
	mpirun -np 4 ./sine
	./dosineplot 

sinefplot: sinef
	mpirun -np 4 ./sinef
	./dosineplot 

clean:
	rm -rf *.o *.~ *.txt $(EXES) *.dat sinef sine
