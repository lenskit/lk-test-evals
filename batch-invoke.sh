#!/bin/sh

hostname
ulimit -v unlimited
ulimit -u 2048
ulimit -s 65536
export MKL_THREADING_LAYER=tbb

exec srun invoke "$@"
