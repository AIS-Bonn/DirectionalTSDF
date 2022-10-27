#!/usr/bin/env bash

git submodule update --init --recursive

NUM_CPUS=$(lscpu | grep "^CPU(s)" | awk {'print $2'})
if [[ -z $NUM_CPUS ]]; then
  NUM_CPUS=8
fi

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PNG=ON ..
make -j${NUM_CPUS}
