#!/bin/bash

if pgrep -x "run_grid.sh"
then
    echo "Running"
else
    echo "Stopped"
fi
