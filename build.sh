#!/usr/bin/env bash

git submodule update --init --recursive

mkdir -p build
cd build
cmake -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release -DWITH_PNG=ON  ..
make -j12 InfiniTAM_cli


