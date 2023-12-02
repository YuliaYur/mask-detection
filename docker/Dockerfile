FROM ubuntu:20.04

# Set arg to avoid interactive prompts
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -yqq && apt upgrade -yqq && apt-get install -yqq \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    subversion \
    cmake \
    libgl1 \
    libglib2.0-0

RUN python3.9 -m pip install --upgrade pip

ENV SETUP_DIR=/setup
WORKDIR ${SETUP_DIR}

ADD requirements.txt ${SETUP_DIR}/requirements.txt
RUN python3.9 -m pip --no-cache-dir install --timeout 20000 -r requirements.txt
RUN python3.9 -m pip --no-cache-dir install dill==0.3.7
RUN python3.9 -m pip --no-cache-dir install dominate==2.8.0

RUN python3.9 -m pip --no-cache-dir install addict==2.4.0 future==0.18.3 lmdb==1.4.1 tb-nightly==2.16.0a20231111 yapf==0.40.2 lpips==0.1.4 gdown==4.7.1
RUN python3.9 -m pip --no-cache-dir install torchaudio==2.1.0
RUN python3.9 -m pip --no-cache-dir install Cython==3.0.5
RUN python3.9 -m pip --no-cache-dir install shapely==2.0.2
