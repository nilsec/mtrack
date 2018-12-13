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
        python-scipy \
        wget \
        curl \
        gzip && \
    rm -rf /var/lib/apt/lists/*


# Install gurobi:
ENV GUROBI_INSTALL /opt/gurobi
ENV GUROBI_HOME $GUROBI_INSTALL/linux64
ENV PATH $PATH:$GUROBI_HOME/bin
ENV LD_LIBRARY_PATH $GUROBI_HOME/lib

ARG GUROBI_MAJOR_VERSION=8.1
ARG GUROBI_VERSION=8.1.0
ARG GUROBI_TAG=810

RUN mkdir -p ${GUROBI_INSTALL}                                                                                      && \
    wget -P /home/gurobi/ http://packages.gurobi.com/${GUROBI_MAJOR_VERSION}/gurobi${GUROBI_VERSION}_linux64.tar.gz && \
    tar xvfz /home/gurobi/gurobi${GUROBI_VERSION}_linux64.tar.gz                                                    && \
    mv /gurobi${GUROBI_TAG}/linux64/ ${GUROBI_INSTALL}                                                              && \
    # Clean up
    rm -rf ${GUROBI_HOME}/docs                             && \
    rm -rf ${GUROBI_HOME}/examples                         && \
    rm -rf ${GUROBI_HOME}/src

# Install graph-tool
RUN echo "deb http://downloads.skewed.de/apt/xenial xenial universe" >> /etc/apt/sources.list && \
    echo "deb-src http://downloads.skewed.de/apt/xenial xenial universe" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --allow-unauthenticated python-graph-tool

# Install pylp
WORKDIR /src/pylp
RUN git clone https://github.com/funkey/pylp.git . && \
    git checkout 51112ba453c361e6980d7c8b24faace2bd7c8064 && \
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
    git submodule update --init && \
    python setup.py install && \
    echo "from skeletopyze import *" > ./skeletopyze/__init__.py
ENV PYTHONPATH /src/skeletopyze:$PYTHONPATH

# Install mtrack
ADD mtrack /src/mtrack
WORKDIR /src/mtrack
RUN pip install -r requirements.txt
ENV PYTHONPATH /src/mtrack:$PYTHONPATH

WORKDIR /src/mtrack
