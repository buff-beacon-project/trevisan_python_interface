FROM ubuntu:xenial as base
RUN apt-get update && apt-get install -y make \
  cmake \
  libgmp3-dev \
  libntl-dev \
  g++ \
  libtercpp-dev \
  libssl-dev \
  libgf2x-dev \
  libtclap-dev \
  libffi-dev \
  wget \
  git

WORKDIR /tmp/python
RUN wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tgz -O - | tar xzvf -
WORKDIR /tmp/python/Python-3.8.5
RUN ./configure && make && make install
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py
RUN wget https://bootstrap.pypa.io/pip/3.5/get-pip.py -O get-pip.py
RUN python3 get-pip.py
RUN rm -rf /tmp/python

FROM base as release
WORKDIR /

RUN mkdir ./trev/
COPY ./trev/ ./trev/
RUN make -C ./trev/

RUN python3 -m pip install pyzmq
RUN python3 -m pip install git+https://github.com/kshalm/zmqhelpers.git

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip --no-cache-dir install -r requirements.txt
COPY ./python .
