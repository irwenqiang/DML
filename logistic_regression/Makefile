lr:lr.o opt_algo.o
	mpicxx -g -O3 -o lr lr.o opt_algo.o
lr.o:lr.cpp lr.h opt_algo.h
	mpicxx -g -O3 -c lr.cpp opt_algo.cpp
clean:
	rm -f *~ lr lr.o
