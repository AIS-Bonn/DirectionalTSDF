#!/usr/bin/env bash

# Check args
if [ "$1" == "-h" ]; then
  echo "usage: ./build.sh IMAGE_NAME"
  return 1
fi

IMAGE_NAME=$1
if [ -z $IMAGE_NAME ]; then
  IMAGE_NAME="dtsdf"
fi

# Get this script's path
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

cd $SCRIPTPATH

# Build the docker image
docker build \
  -t $IMAGE_NAME \
  --rm \
  --build-arg user=$USER\
  --build-arg uid=$UID\
  --build-arg gid=$(id -g)\
  --build-arg home=$HOME\
  --build-arg shell=/bin/bash \
  -f Dockerfile .
#  --no-cache \
