#!/bin/bash

#train code
train:main.o opt_algo.o 
	mpicxx -o train main.o opt_algo.o -L/opt/OpenBLAS/lib -lopenblas -lpthread

main.o: main.cpp 
	mpicxx -I /opt/OpenBLAS/include -I ./include -c main.cpp

opt_algo.o: opt_algo.cpp  
	mpicxx -I /opt/OpenBLAS/include/ -I ./include -c opt_algo.cpp

#predict code
predict:predict.o
	mpicxx -g -o predict -lpthread -L/opt/OpenBLAS/lib -lopenblas predict.o

predict.o:predict.cpp
	mpicxx -I /opt/OpenBLAS/include/ -c predict.cpp

#make train uttest
train_ut:train_uttest.o
	#mpicxx -o train_ut train_uttest.o -L/home/wangxiaoshu/document/gtest/lib -lgtest
	mpicxx -o train_ut train_uttest.o -L ./lib-lgtest

train_uttest.o: train_uttest.cpp
	mpicxx  -I ./include -c train_uttest.cpp
clean:
	rm -f *~ train predict *.o
