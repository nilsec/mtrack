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

WORKDIR /src/pylp
RUN git clone https://github.com/funkey/pylp.git . && \
    git checkout 08711e9988be3e2cf71f9ffbbb707f83144653e5 && \
    git submodule update --init && \
    python setup.py install
ENV PYTHONPATH /src/pylp:$PYTHONPATH
