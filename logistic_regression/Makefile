lr:opt_algo.o
	mpicxx -g -lpthread -O3 -o lr lr.o opt_algo.o 
opt_algo.o:lr.cpp opt_algo.cpp opt_algo.h
	mpicxx -g -O3 -c lr.cpp opt_algo.cpp
clean:
	rm -f *~ lr lr.o
