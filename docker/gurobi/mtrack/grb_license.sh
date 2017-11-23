#!/bin/sh

license=/src/gurobi/gurobi.lic
if [ -f $license ]; then
    echo "License found, done!"
    echo $license
    bash $1
else 
    echo "No license, provided. Exit."
    echo $license
    bash $1
fi
