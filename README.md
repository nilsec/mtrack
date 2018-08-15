# MTrack

Automatic extraction of microtubules in electron microscopy volumes of neural tissue.

## Install


### Using Docker [Recommended]

1. Install docker: 
    
    For installation instructions see https://docs.docker.com/install/linux/docker-ce/ubuntu/

2. Build (or pull) the image:
    
    ```    
    git clone https://github.com/nilsec/mtrack.git

    cd mtrack/docker

    make
    ```

    or

    ```
    docker pull nilsec/mtrack
    ```

### Using Singularity
```
singularity pull docker://nilsec/mtrack
```

### Build from source

1. See mtrack/docker/mtrack/Dockerfile for requirements, package version numbers and installation instructions.

2. Once all requirements are fulfilled install with:
    ```
    python setup.py install
    ```

## Usage
1. Start an mtrack docker container with mongodb instance:

    ```
    cd mtrack/docker

    ./start_container
    ```

1.2 [Singularity only]:
```
singularity shell docker://nilsec/mtrack

```

2. The container created in 1. is now ready to be used. Your home directory is mounted into the countainer such that it is now possible to work interactively inside the container as if a local installation was performed. Use the container id provided by ./start_container to access a bash shell inside the container:
    
    ```
    docker exec -it <container_id> /bin/bash
    ```

2.2 [Singularity only]:
Start mongod instance in the background
```
mongod --fork --logpath <logpath> --dbpath <dbsavedir>
```

3. A typical workflow would require training of a microtubule RFC (it is better to train one for perpendicular appearing mt's and another for elongated ones) via Ilastik (http://ilastik.org/), tracing of a small validation set and then a grid search over the solve parameters:

```python
    from mtrack.mt_utils.grid_search import generate_grid, run_grid
    from mtrack.evaluation import EvaluationParser

    parameter_dict = {"evaluate": [True],
                      "tracing_file": ["Validation tracing path"],
                      ...,
                     "start_edge_prior": [140, 150, 160],
                     "selection_cost": [-70.0, -80.0, -90.0],
                     ...,
                     "ilastik_source_dir": ["Your Ilastik source dir"],
                     "ilastik_project_perp": ["Perpendicular microtubule Ilastik classifier"],
                     "ilastik_project_par": ["Parallel microtubule Ilastik classifier"],
                     ...
                     }

    generate_grid(parameter_dict, "./grid_search")
    run_grid("./grid_search", n_workers=8, skip_condition=lambda cfg: False)
```

Note that ilastik probability map generation is currently not supported within the docker container. Thus it is recommended to extract the probability maps before, outside of docker, and simply pass their locations to ```track``` or ```generate_grid``` via the config file or the ```parameter_dict``` respectively. 

4. Set best performing config parameters in mtrack/config.ini or write your own config file using the provided config as a template. Reconstruct microtubules via:
    ```python
    from mtrack import track

    track(cfg_path)
    ```
