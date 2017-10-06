#!/bin/bash
until python ./run_grid.py; do
    echo "Crashed with exit code $?. Restarting..."
    sleep 1
done
