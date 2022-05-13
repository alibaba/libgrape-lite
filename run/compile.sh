#!/bin/sh

#mpic++ -std=c++14 -O3 ../src/apps/$1.cpp -o $1

mpic++ -std=c++14 -O3 -fopenmp ../src/alg/$1.cpp -o $1
