#!/bin/sh

./venv-seq/bin/invoke time-train --type seq --data ml-100k
./venv-seq/bin/invoke time-train --type seq --data ml-1m
./venv-omp/bin/invoke time-train --type openmp --data ml-100k
./venv-omp/bin/invoke time-train --type openmp --data ml-1m

for th in 1 2 3 4 5 6 7 8 12 16 20 24 28; do
    echo "Running with $th threads"
    env OMP_NUM_THREADS="$th" ./venv-omp/bin/invoke time-train --type openmp --data ml-100k --threads $th
    env OMP_NUM_THREADS="$th" ./venv-omp/bin/invoke time-train --type openmp --data ml-1m --threads $th
done
