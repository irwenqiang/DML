#!/bin/bash

SRCPATH=-I./src -I./include 
#train code
train:main.o opt_algo.o 
	mpicxx -g -o train main.o opt_algo.o -L/opt/OpenBLAS/lib -lopenblas -lpthread --std=c++11

main.o: main.cpp 
	mpicxx -I /opt/OpenBLAS/include -I ./include -c main.cpp 

opt_algo.o: opt_algo.cpp
	mpicxx -I /opt/OpenBLAS/include/ -I ./include -c opt_algo.cpp --std=c++11

#predict code
predict:predict.o
	mpicxx -g -o predict -lpthread -L/opt/OpenBLAS/lib -lopenblas predict.o

predict.o:predict.cpp
	mpicxx -I /opt/OpenBLAS/include/ -c predict.cpp

#make train uttest
train_ut:train_uttest.o opt_algo.o
	mpicxx -o train_ut train_uttest.o opt_algo.o -L /opt/OpenBLAS/lib -lopenblas -lpthread -L ./lib -lgtest

train_uttest.o: train_uttest.cpp
	mpicxx  -I ./include -c train_uttest.cpp
clean:
	rm -f *~ train predict *.o
