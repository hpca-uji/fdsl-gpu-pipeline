#!/bin/bash

# # vatom
nvcc -rdc=true vatom.cu -o prgvatom -lcudadevrt
./prgvatom
rm prgvatom

# profile
nvcc -rdc=true profile.cu -o prgprofile -lcudadevrt
nsys profile ./prgprofile
rm prgprofile

# single kernel (for usage in pytorch integration)
nvcc -rdc=true singlekernel.cu -o prgsinglekernel -lcudadevrt
./prgsinglekernel
rm prgsinglekernel