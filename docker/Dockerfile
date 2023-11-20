FROM nvidia/cuda:11.7.1-base-ubuntu20.04

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
        build-essential \
        cmake \
        ffmpeg \
        git \
        python-is-python3 \
        python3-dev \
        python3-pip \
        && \
    rm -rf /var/lib/apt/lists/*
 
COPY docker/requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade --requirement requirements.txt
RUN python3 -m pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

COPY docker/ext_packages /docker/ext_packages
RUN python3 /docker/ext_packages/install_upstream_python_packages.py

COPY . /gatr/
RUN python3 -m pip install --no-cache-dir /gatr && rm -rf /gatr
