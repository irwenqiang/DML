lr:opt_algo.o
	mpicxx -g -o lr -lpthread -L/opt/OpenBLAS/lib -lopenblas opt_algo.o lr.o
opt_algo.o:main.cpp opt_algo.cpp
	mpicxx -I /opt/OpenBLAS/include -c main.cpp opt_algo.cpp
clean:
	rm -f *~ lr lr.o
