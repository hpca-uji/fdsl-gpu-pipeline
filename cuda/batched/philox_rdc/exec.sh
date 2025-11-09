#!/bin/bash

# vatom
nvcc -rdc=true vatom.cu -o prgvatom -lcudadevrt
nsys profile --trace cuda ./prgvatom
rm prgvatom

#vatom with classes
nvcc -rdc=true multiclass.cu -o prgmulticlass -lcudadevrt
./prgmulticlass
rm prgmulticlass

# profile
nvcc -rdc=true profile.cu -o prgprofile -lcudadevrt
./prgprofile
rm prgprofile