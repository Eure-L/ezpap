#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

sizes = [2176]
itt = 2000
nbrun = 1

for s in sizes:
    options = {}
    ompenv = {}
    options["-k "] = ["life"]
    options["-s "] = [s]
    options["-i "] = [itt]
    options["-a "] = ["otca_off"]

    #parametres pour nos runs
    options["-tw "] = [32,64,128]
    options["-v "] = ["lazybtmpvec"]
    
    #parametres de precompileur 
    ompenv["OMP_NUM_THREADS="] = [1] + list(range(2, 47,2))
    ompenv["OMP_PLACES="] = ["cores"]
    
    print(ompenv, options, nbrun)
    execute('./run ', ompenv, options, nbrun, verbose=True, easyPath=".")

    

