#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

sizes = [4096]
#sizes = ["2176 -a otca_off","6208 -a meta3x3 "]
itt = 1000
nbrun = 20

for s in sizes:
    options = {}
    ompenv = {}
    options["-k "] = ["life"]
    
    options["-i "] = [itt]
    options["-a "] = ["random"]
    options["-s "] = [s]
    options["-tw "]=[32,64,128,256]
    options["-th "]=[32,64,128,256]
    options["-lb "] = ["bestTile"]
#parametres pour nos runs
    options["-v "] = ["bitbrd"]
    
    
    #parametres de precompileur 
    ompenv["OMP_NUM_THREADS="] = [22] #+ list(range(2, 25,2))
    ompenv["OMP_PLACES="] = ["cores"]
    
    print(ompenv, options, nbrun)
    execute('TILEX=32 TILEY=32 ./run -n ', ompenv, options, nbrun, verbose=True, easyPath=".")

    itt = itt/4
    

