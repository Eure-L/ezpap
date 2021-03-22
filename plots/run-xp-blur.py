#!/usr/bin/env python3
from graphTools import *
from expTools import *
import os

easyspap_options = {}
easyspap_options["--kernel "] = ["blur"]
easyspap_options["--iterations "] = [9]
easyspap_options["--variant "] = ["tiled", "tiled_opt","tiled_omp","omp_tiled"]
easyspap_options["--tile-size "] = [8,16,32,64,128,256]
easyspap_options["--size "] = [1024]

omp_icv = {}  # OpenMP Internal Control Variables
omp_icv["OMP_NUM_THREADS="] = [1] + list(range(2, 25, 2))
omp_icv["OMP_SCHEDULE="] = ["static"]
omp_icv["OMP_PLACES="] = ["cores"]

execute('./run', omp_icv, easyspap_options, nbrun=1)
