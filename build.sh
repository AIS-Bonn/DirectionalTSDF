#!/usr/bin/env bash

git submodule update --init --recursive

mkdir -p build
cd build
#cmake -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang-8 -DCMAKE_CXX_COMPILER=/usr/bin/clang++-8 -DCMAKE_C_COMPILER=/usr/bin/clang-8 -DCMAKE_CUDA_COMPILER:PATH=/usr/bin/nvcc -DCMAKE_BUILD_TYPE=Release -DWITH_PNG=ON ..
cmake -DCMAKE_CUDA_COMPILER:PATH=/usr/bin/nvcc -DCMAKE_BUILD_TYPE=Release -DWITH_PNG=ON ..
make -j8
