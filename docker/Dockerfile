FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

WORKDIR /app
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
        git \
        less \
        python-is-python3 \
        python3 \
        python3-pip \
        tree \
        vim \
	&& \
    rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html && \
    python3 -m pip install --no-cache-dir --upgrade --requirement requirements.txt

COPY . /app
