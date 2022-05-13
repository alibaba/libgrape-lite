#!/bin/sh


#mpic++ -std=c++14 -O3  ../src/flash.cpp -o flash

mpic++ -std=c++14 -O3 -fopenmp ../src/flash.cpp -o flash