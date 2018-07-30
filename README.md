# MTrack

Automatic extraction of microtubules in electron microscopy volumes of neural tissue.

## Install


### Using Docker [Recommended]

1. Install docker: 
    
    For installation instructions see https://docs.docker.com/install/linux/docker-ce/ubuntu/

2. Build the image:
    
    ```    
    git clone https://github.com/nilsec/mtrack.git

    cd mtrack/docker

    make
    ```

### Build from source

1. See mtrack/docker/mtrack/Dockerfile for requirements, package version numbers and installation instructions.

2. Once all requirements all fulfilled build an install with:
    ```
    python setup.py install
    ```

## Usage
### Using Docker:

1. Start an mtrack docker container with mongodb instance:
    ```
    cd mtrack/docker

    ./start_container
    ```
2. The container created in 1. is now ready to be used. Your home directory is mounted into the countainer such that it is now possible to work interactively inside the container as if a local installation was performed. Use the container id provided by ./start_container to access a bash shell inside the container:
    
    ```
    docker exec -it <container_id> /bin/bash
    ```

3. Set config parameters as appropriate in mtrack/config.ini or write your own config file using the provided config as a template.

4. Track:
    ```python
    from mtrack import track

    track(cfg_path)
