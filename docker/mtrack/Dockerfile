FROM mongo:rc-xenial
LABEL maintainer nilsec@ini.ethz.ch

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        mercurial \
        libboost-all-dev \
        libhdf5-serial-dev \
        libtiff5-dev \
        libpng12-dev \
        libfftw3-dev \        
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

# Install graph-tool
RUN echo "deb http://downloads.skewed.de/apt/xenial xenial universe" >> /etc/apt/sources.list && \
    echo "deb-src http://downloads.skewed.de/apt/xenial xenial universe" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --allow-unauthenticated python-graph-tool

# Install pylp
WORKDIR /src/pylp
RUN git clone https://github.com/funkey/pylp.git . && \
    git checkout 08711e9988be3e2cf71f9ffbbb707f83144653e5 && \
    git submodule update --init
RUN SOURCE_DIR=/src/pylp python setup.py install
ENV PYTHONPATH /src/pylp/pylp:$PYTHONPATH

# Install vigra:
WORKDIR /src/vigra
RUN pip install nose && \
    git clone git://github.com/ukoethe/vigra.git . && \
    git checkout fb427440da8c42f96e14ebb60f7f22bdf0b7b1b2 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make check && \
    make install

# Install skeletopyze
WORKDIR /src/skeletopyze
RUN git clone https://github.com/funkey/skeletopyze.git . && \
    git checkout 6146901a43506bb5d50feff60d2b538e6e9f6f58 && \
    git submodule update --init && \
    python setup.py install && \
    echo "from skeletopyze import *" > ./skeletopyze/__init__.py
ENV PYTHONPATH /src/skeletopyze:$PYTHONPATH

# Install mtrack
ADD mtrack /src/mtrack
WORKDIR /src/mtrack
RUN pip install -r requirements.txt
ENV PYTHONPATH /src/mtrack:$PYTHONPATH
ENV MUST_USE_SCIP True

WORKDIR /src/mtrack
