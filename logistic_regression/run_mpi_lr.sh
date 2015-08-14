#!/bin/bash

mpiexec -mca btl ^openib -np 3 ./train 
