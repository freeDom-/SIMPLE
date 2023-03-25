FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime as base

RUN apt-get update
RUN apt-get -y install ssh
RUN apt-get -y install python3-pip
RUN apt-get -y install htop
RUN apt-get -y install libpq-dev

RUN apt-get -y install cmake python3-dev zlib1g-dev libgl1-mesa-dev glibc-source

RUN pip3 install --upgrade pip

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
