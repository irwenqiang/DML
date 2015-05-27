lr:lr.o
	mpicxx  -fopenmp -g -O3 -o lr lr.o
lr.o:lr.cpp lr.h
	mpicxx -fopenmp -g -O3 -c lr.cpp
clean:
	rm -f *~ lr lr.o
