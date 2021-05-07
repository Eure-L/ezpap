#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

sizes = [512,1024,2048,4096,8192]
itt = 25600
nbrun = 1

for s in sizes:
    options = {}
    ompenv = {}
    options["-k "] = ["life"]
    options["-s "] = [s]
    options["-i "] = [itt]
    options["-a "] = ["random"]

    #parametres pour nos runs
    options["-v "] = ["omp_hybrid","omp","lazybtmpvec"]
    
    #parametres de precompileur 
    ompenv["OMP_NUM_THREADS="] = [1] + list(range(2, 25,2))
    ompenv["OMP_PLACES="] = ["cores"]
    
    print(ompenv, options, nbrun)
    execute('TILEX=32 TILEY=32 ./run -o', ompenv, options, nbrun, verbose=True, easyPath=".")

    itt = itt/4
    

