#!/bin/bash

SRCPATH=-I . -I ./src -I ./include -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib
CC=mpicxx
OPT=-g -Wall --std=c++11 -O3
VPATH=./src/ ./include/
INCLUDE=-I ./src -I ./include
#train code
train:main.o opt_algo.o 
	${CC} ${OPT} -o train main.o opt_algo.o ${SRCPATH} -lopenblas -lpthread

main.o: main.cpp 
	${CC} ${OPT} ${SRCPATH} -c ./src/main.cpp

opt_algo.o: opt_algo.cpp
	${CC} ${OPT} ${SRCPATH} -c ./src/opt_algo.cpp

#predict code
predict:predict.o
	${CC} ${OPT} -o predict -lpthread ${SRCPATH} -lopenblas predict.o

predict.o:predict.cpp
	${CC} ${OPT} ${SRCPATH} -c ./src/predict.cpp

#make train uttest
train_ut:train_uttest.o opt_algo.o
	${CC} ${OPT} -o train_ut train_uttest.o opt_algo.o -L /opt/OpenBLAS/lib -lopenblas -lpthread -L ./lib -lgtest

train_uttest.o: train_uttest.cpp
	${CC} ${OPT} -c ${SRCPATH} ./src/train_uttest.cpp
.PHONY:clean
	clean
clean:
	rm -f *~ train predict *.o
