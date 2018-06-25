FROM mongo:rc-xenial
LABEL maintainer nilsec@ini.ethz.ch

# Set environment variables for gurobi
ENV GUROBI_INSTALL /opt/gurobi
ENV GUROBI_HOME $GUROBI_INSTALL/linux64
ENV PATH $PATH:$GUROBI_HOME/bin
ENV LD_LIBRARY_PATH ${GUROBI_HOME}/lib

ARG GUROBI_MAJOR_VERSION=6.5
ARG GUROBI_VERSION=6.5.2

# install gurobi
RUN apt-get update &&               \
    apt-get install -y wget &&      \
    rm -rf /var/lib/apt/lists/* &&  \
    mkdir -p ${GUROBI_INSTALL} &&                                                                                      \
    wget -P /home/gurobi/ http://packages.gurobi.com/${GUROBI_MAJOR_VERSION}/gurobi${GUROBI_VERSION}_linux64.tar.gz && \
    tar xvfz /home/gurobi/gurobi${GUROBI_VERSION}_linux64.tar.gz  &&                                                   \
    mv /gurobi652/linux64/ ${GUROBI_INSTALL} &&                                                                        \
    mkdir ${GUROBI_INSTALL}/linux64/scripts &&                                                                         \
    rm -rf ${GUROBI_HOME}/docs &&                 \
    rm -rf ${GUROBI_HOME}/examples &&             \
    rm -rf ${GUROBI_HOME}/src &&                  \
    rm -rf /var/cache/apk/* &&                    \
    rm -rf /tmp/* &&                              \
    rm -rf /var/log/* &&                          \
    rm -rf /gurobi652 &&                          \
    rm /home/gurobi/gurobi${GUROBI_VERSION}_linux64.tar.gz &&   \
    apt-get clean
