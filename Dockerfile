# syntax=docker/dockerfile:1
#FROM amd64/ubuntu:xenial
FROM ubuntu:xenial
RUN apt-get update

RUN apt-get install -y make \
&& apt-get install -y libgmp3-dev \
&& apt-get install -y libntl-dev \
&& apt-get install -y g++ \
&& apt-get install -y libtercpp-dev \
&& apt-get install -y git \
&& apt-get install -y libssl-dev \
&& apt-get install -y libgf2x-dev \
&& apt-get install -y libtclap-dev \
&& apt-get install -y libffi-dev \
&& apt-get install -y curl \
&& apt-get install -y wget

RUN wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tgz
RUN tar xzvf Python-3.8.5.tgz
WORKDIR Python-3.8.5
RUN ./configure
RUN make
RUN make install

WORKDIR /
RUN mkdir ./trev/
COPY ./trev/ ./trev/
RUN make -C ./trev/

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py
RUN python3 get-pip.py
# WORKDIR /app


RUN pip3 install --upgrade pip
RUN pip3 install pyzmq
RUN python3 -m pip install git+https://github.com/kshalm/zmqhelpers.git
# RUN pip3 install numpy

WORKDIR /app
COPY . .
# RUN python3 -m pip install --upgrade pip && \
# pip3 --no-cache-dir install -r requirements.txt
RUN python3 -m pip --no-cache-dir install -r requirements.txt
