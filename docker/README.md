# Docker

This directory contains all necessary files to create a mtrack Docker image. Right now the Docker install only supports the Scip solver as usage of Gurobi inside Docker containers is not covered by the free academic license.

In order to build the image locally run:

    ```    
    make
    ```

In order to start up a container run:

    ```
    ./start_container
    ```

Interactively work with the container via:

    ```
    docker exec -it <container_id> /bin/bash
    ```
