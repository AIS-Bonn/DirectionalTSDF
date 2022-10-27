# syntax=docker/dockerfile:1
FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

# switch to non interactive https://github.com/phusion/baseimage-docker/issues/319
ENV DEBIAN_FRONTEND noninteractive

# standard tools
RUN apt-get -y update && apt-get install -y --no-install-recommends \
  apt-utils \
  cmake \
  htop \
  less \
  sudo \
  tmux \
  vim

#fixing locales in docker https://daten-und-bass.io/blog/fixing-missing-locale-setting-in-ubuntu-docker-image/
RUN sudo apt-get -y update \
    && sudo apt-get install -y locales locales-all \
    && sudo sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && sudo dpkg-reconfigure --frontend=noninteractive locales \
    && sudo update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
RUN sudo locale-gen

# Necessary for building project
RUN sudo apt-get update \
  && sudo apt-get install -y clang cmake git libeigen3-dev freeglut3-dev zlib1g-dev libpng-dev libyaml-cpp-dev

# Clean caches etc to reduce image size
COPY cleanup.sh ./
RUN chmod +x ./cleanup.sh && sh ./cleanup.sh

# Arguments passed from build.sh
# Setup same user inside container, so alterations inside the code don't cause any permission problems
ARG user
ARG uid
ARG gid
ARG home
ARG shell

# Create user inside docker image
RUN groupadd --gid ${gid} ${user} \
  && useradd --uid ${uid} --gid ${gid} --shell ${shell} --home-dir ${home} ${user} \
  && echo "${user} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${user}" \
  && chmod 0440 "/etc/sudoers.d/${user}"
RUN mkdir -p ${home} \
  && chown -R ${user} ${home}

# Set the user in the appropriate groups
RUN usermod -a -G dialout ${user} \
  && usermod -a -G video ${user} \
  && usermod -a -G audio ${user} \
  && usermod -a -G plugdev ${user}

# set default user and working directory
USER ${user}
WORKDIR ${home}/code

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# build with clang
ENV CC clang
ENV CXX clang++
