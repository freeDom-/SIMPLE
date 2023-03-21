#FROM ubuntu:bionic-20200219 as base
FROM tensorflow/tensorflow:1.15.5-gpu-py3-jupyter as base

RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update
RUN apt-get -y install ssh
RUN apt-get -y install python3-pip
RUN apt-get -y install htop
RUN apt-get -y install libpq-dev

RUN apt-get update 
RUN apt-get -y install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-dev

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

RUN useradd -ms /bin/bash selfplay
USER selfplay
ENV PATH="/home/selfplay/.local/bin:${PATH}"
WORKDIR /app


COPY --chown=selfplay:selfplay ./app/requirements.txt /app
RUN pip3 install -r /app/requirements.txt

COPY --chown=selfplay:selfplay ./app .

ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV export LANG=C.UTF-8
