#!/bin/sh

mpirun -n $1 ./flash format $2 $3 $4
