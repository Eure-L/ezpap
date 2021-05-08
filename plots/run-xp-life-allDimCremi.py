#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

sizes = [512,1024,2048,4096,8192]
#sizes = ["2176 -a otca_off -lb loadstatic","6208 -a meta3x3 -lb loadstatic"]
itt = 25600
nbrun = 10

for s in sizes:
    options = {}
    ompenv = {}
    options["-k "] = ["life"]
    
    options["-i "] = [itt]
    options["-a "] = ["random"]
    options["-s "] = [s]
    #options["-lb "] = [""]
#parametres pour nos runs
    options["-v "] = ["bitbrd"]
    
    #parametres de precompileur 
    ompenv["OMP_NUM_THREADS="] = [1] + list(range(2, 25,2))
    ompenv["OMP_PLACES="] = ["cores"]
    
    print(ompenv, options, nbrun)
    execute('TILEX=32 TILEY=32 ./run -n', ompenv, options, nbrun, verbose=True, easyPath=".")

    itt = itt/4
    

