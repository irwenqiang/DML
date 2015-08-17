#!/bin/bash

#train code
train:main.o opt_algo.o 
	mpicxx -o train main.o opt_algo.o -L/opt/OpenBLAS/lib -lopenblas -lpthread

main.o: src/main.cpp 
	mpicxx -I /opt/OpenBLAS/include -I ./include -c src/main.cpp

opt_algo.o: src/opt_algo.cpp  
	mpicxx -I /opt/OpenBLAS/include/ -I ./include -c src/opt_algo.cpp

#predict code
predict: predict.o
	mpicxx -g -o predict -lpthread -L/opt/OpenBLAS/lib -lopenblas predict.o

predict.o: src/predict.cpp
	mpicxx -I /opt/OpenBLAS/include/ -c src/predict.cpp

#make train uttest
train_ut: train_uttest.o opt_algo.o
	mpicxx -o train_ut train_uttest.o opt_algo.o -L /opt/OpenBLAS/lib -lopenblas -lpthread -L ./lib -lgtest

train_uttest.o: src/train_uttest.cpp
	mpicxx  -I ./include -c src/train_uttest.cpp
clean:
	rm -f *~ train predict *.o
