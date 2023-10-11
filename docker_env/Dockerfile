FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install common package
# coreutils for md5sum 
ENV TZ=Asia/Hong_Kong
RUN apt-get update && apt-get install -y tzdata
RUN apt-get update && apt-get install -y \
  wget \
  coreutils \
  git-all \
  curl

# Install python 3.9
# https://stackoverflow.com/a/65644846/11693034
RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  apt-get install -y python3.9 python3.9-venv && \
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
  apt-get install -y python3.9-distutils && \
  python3.9 get-pip.py && \
  rm get-pip.py

ENV PATH="${PATH}:/root/.local/bin"

# RUN pip install --upgrade pip && pip install \
#   jax[cuda11_pip] jaxlib flax[all] \
#   tensorflow

# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

WORKDIR /workspace
