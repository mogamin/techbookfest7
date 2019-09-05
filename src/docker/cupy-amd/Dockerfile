FROM ubuntu:18.04

MAINTAINER Shunta Saito <shunta.saito@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg && \
  curl -sL http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
  sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list' && \
  apt-get update && apt-get install -y --no-install-recommends \
  sudo \
  kmod \
  libelf1 \
  libnuma-dev \
  build-essential \
  git \
  python3 \
  python3-dev \
  python3-pip \
  python3-setuptools \
  python3-wheel \
  python3-tk \
  hipblas \
  hipsparse \
  rocrand \
  vim-nox \
  cmake-curses-gui \
  rocm-dev \
  rocm-smi && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

ARG HCC_TARGET=gfx900
ENV HCC_AMDGPU_TARGET=${HCC_TARGET}
ENV ROCM_HOME=/opt/rocm
ENV __HIP_PLATFORM_HCC__=""
ENV CUPY_INSTALL_USE_HIP=1
ENV PATH=$ROCM_HOME/bin:$PATH
ENV CUDA_PATH=/opt/rocm

RUN pip3 install cython numpy chainer==7.0.0b2 matplotlib six fastrlock

ENV CFLAGS="-I/opt/rocm/include -I/opt/rocm/hiprand/include -I/opt/rocm/rocrand/include"
ENV LDFLAGS="-L/opt/rocm/lib -L/opt/rocm/hiprand/lib -L/opt/rocm/rocrand/lib -L/opt/rocm/hip/lib"

RUN git clone https://github.com/okuta/cupy -b support-hip && \
    cd cupy && pip3 install -e . -vvv

RUN git clone https://github.com/chainer/chainer -b v7.0.0b2 --depth=1