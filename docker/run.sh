#!/usr/bin/env bash

# Get this script's path
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

ROOT=$(dirname ${SCRIPTPATH})

#  -v ${HOME}/datasets:${HOME}/datasets \
docker run \
  -v ${ROOT}:${HOME}/code \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --device=/dev/dri:/dev/dri \
  --net=host \
  -ti \
  --rm \
  --gpus all \
  dtsdf bash
