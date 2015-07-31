#!/bin/bash

train:main.o opt_algo.o utils.o  
	mpicxx -g -o train -lpthread -L/opt/OpenBLAS/lib -lopenblas main.o

main.o: main.cpp opt_algo.h utils.h utils.cpp 
	mpicxx -I /opt/OpenBLAS/include -c main.cpp

opt_algo.o: opt_algo.cpp opt_algo.h 
	mpicxx -I /opt/OpenBLAS/include/ -c opt_algo.cpp

utils.o: utils.cpp utils.h
	mpicxx -I /opt/OpenBLAS/include/ -c utils.cpp

predict:predict.o
	mpicxx -g -o predict -lpthread -L/opt/OpenBLAS/lib -lopenblas predict.o

predict.o:predict.cpp predict.h
	mpicxx -I /opt/OpenBLAS/include/ -c predict.cpp

clean:
	rm -f *~ train predict *.o
