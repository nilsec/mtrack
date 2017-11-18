FROM gurobi
LABEL maintainer nilsec@ini.ethz.ch

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libboost-all-dev \
        libhdf5-serial-dev \
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
    git submodule update --init && \
    python setup.py install
ENV PYTHONPATH /src/pylp:$PYTHONPATH

# Install vigra:
WORKDIR /src/vigra
RUN git clone git://github.com/ukoethe/vigra.git . && \
    git checkout fb427440da8c42f96e14ebb60f7f22bdf0b7b1b2 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make check && \
    make install

# Install skeletopyze
WORKDIR /src/pylp
RUN git clone https://github.com/funkey/skeletopyze.git . && \
    git checkout 6146901a43506bb5d50feff60d2b538e6e9f6f58 && \
    git submodule update --init && \
    python setup.py install
ENV PYTHONPATH /src/pylp:$PYTHONPATH

# Install mtrack
ADD mtrack /src/mtrack/mtrack
ADD requirements.txt /src/mtrack/requirements.txt
ADD setup.py /src/mtrack/setup.py
WORKDIR /src/mtrack
RUN pip install -r requirements.txt
RUN python setup.py install
ENV PYTHONPATH /src/mtrack:$PYTHONPATH
