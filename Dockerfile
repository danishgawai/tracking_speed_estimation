FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get upgrade -y && apt-get install -y \
    python3 \
    python3-pip \
    libopencv-core-dev \
    ffmpeg \
    nano \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

ADD . /app
WORKDIR /app
# RUN pip3 install -r requirements.txt
# Install specific versions of torch and ultralytics to ensure compatibility
RUN pip3 install torch ultralytics openpyxl pandas opencv-python-headless lap cython_bbox
