#!/bin/bash

# Install dependencies
brew install libomp
brew install llvm

# Set environment variables
LLVM_PATH=$(brew --prefix llvm)
LIBOMP_PATH=$(brew --prefix libomp)

echo "CC=${LLVM_PATH}/bin/clang" >> $GITHUB_ENV
echo "CXX=${LLVM_PATH}/bin/clang++" >> $GITHUB_ENV
echo "CFLAGS=-Wno-unused-result -Wsign-compare -Wunreachable-code -fno-common -dynamic -I${LIBOMP_PATH}/include -Xpreprocessor -fopenmp" >> $GITHUB_ENV
echo "CXXFLAGS=-Wno-unused-result -Wsign-compare -Wunreachable-code -fno-common -dynamic -I${LIBOMP_PATH}/include -Xpreprocessor -fopenmp" >> $GITHUB_ENV
echo "LDFLAGS=-L${LIBOMP_PATH}/lib -lomp" >> $GITHUB_ENV
echo "${LLVM_PATH}/bin" >> $GITHUB_PATH
