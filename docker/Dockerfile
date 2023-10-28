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
