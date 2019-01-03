#!/bin/sh

rm build/timing.csv

for th in 1 2 4 8 10 14 16 20 24 28; do
    echo "Running with $th threads"
    env NUMBA_NUM_THREADS="$th" invoke time-train --type openmp --data ml-1m --algorithm item-item --threads $th -n 5
    # for mklth in 1 2 4 8; do
    #     echo "Running with $th threads & $mklth mkl"
    #     env NUMBA_NUM_THREADS="$th" MKL_NUM_THREADS="$mklth" \
    #         invoke time-train --type openmp --data ml-1m --algorithm als \
    #         --threads $th --mkl-threads $mklth -n 5
    # done
done
