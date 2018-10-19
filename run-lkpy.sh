#!/bin/sh

hostname
ulimit -v unlimited
ulimit -u 512
ulimit -s 65536
ulimit -a

exec invoke "$@"
