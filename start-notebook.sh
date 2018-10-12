#!/bin/sh

hostname
ulimit -v unlimited
ulimit -a

exec jupyter notebook
