FROM ubuntu:18.04 as base
SHELL ["/bin/bash", "-c"]

# Install base utilities
RUN apt-get update
RUN apt-get install -y wget ssh htop libpq-dev
RUN apt-get install -y cmake libopenmpi-dev zlib1g-dev python3-dev libgl1-mesa-dev
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH="/usr/local/miniconda/bin:${PATH}"
ARG PATH="/usr/local/miniconda/bin:${PATH}"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /usr/local/miniconda
RUN conda init bash
RUN conda update -y conda

# Setup environment
RUN conda create -y -n simple python=3.7 pip
RUN conda install -y -n simple tensorflow-gpu=1.15
RUN conda run -n simple pip install --upgrade pip setuptools wheel

RUN useradd -ms /bin/bash selfplay
USER selfplay
ENV PATH="/home/selfplay/.local/bin:${PATH}"
WORKDIR /app

COPY --chown=selfplay:selfplay ./app/requirements.txt /app
RUN conda run -n simple pip install -r /app/requirements.txt

COPY --chown=selfplay:selfplay ./app .

ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV export LANG=C.UTF-8

RUN echo "conda activate simple" >> ~/.bashrc
