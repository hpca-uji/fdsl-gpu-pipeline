#!/bin/bash

# vatom
nvcc -rdc=true vatom.cu -o prgvatom -lcudadevrt
./prgvatom
rm prgvatom

# profile
nvcc -rdc=true profile.cu -o prgprofile -lcudadevrt
./prgprofile
rm prgprofile