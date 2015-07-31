#!/bin/bash

mpiexec -mca btl ^openib -np 4 ./train 
