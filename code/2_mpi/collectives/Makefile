all: scatter 
include ../Makefile.inc

scatter: scatter.c
	${MPICC} ${CFLAGS} -o $@ $<

gather: gather.c	
	 ${MPICC} ${CFLAGS} -o $@ $<

clean:
	rm -rf scatter
	rm -rf *.o
	rm -rf *~
