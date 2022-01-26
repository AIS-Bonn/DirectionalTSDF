#!/usr/bin/env bash

git submodule update --init --recursive

mkdir -p build
cd build
cmake -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang-9 -DCMAKE_CXX_COMPILER=/usr/bin/clang++-9 -DCMAKE_C_COMPILER=/usr/bin/clang-9 -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release -DWITH_PNG=ON  ..
make -j12 InfiniTAM


