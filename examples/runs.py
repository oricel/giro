import pandas as pd
import numpy as np

import functools
import time
import math

import pickle
import datetime

import networkx as nx
import matplotlib.pyplot as plt
import collections

import random 
import networkx as nx
from networkx.utils import powerlaw_sequence

import collections

from awalker import randomcell

from sys import argv

if __name__ == '__main__':
    Nlist = [500]
    max_iter = 1500
    ystarlist = [0,10]
    deltatlist = [1,3,4,4.5,5,5.5,6,10,15]
    
    seed = int(argv[1])
  
    np.random.seed(seed=seed)
    convergence = []
    n_runs = 500
    for i in range(0,n_runs):
        try:
            for j in range(0,len(Nlist)):
                for d in range(0,len(deltatlist)):
                    for y in range(0, len(ystarlist)):
                        cell = randomcell.Cell(Nlist[j])    
                        print "run=" + str(i) + " with " + "nodes=" + str(Nlist[j]) + ", max iter=" + str(max_iter) + ", ystar=" +  str(ystarlist[y]) + ", deltat=" + str(deltatlist[d])
                        walk = randomcell.Walk(cell, max_iter=max_iter,
                                               ystar = ystarlist[y],
                                               deltat = deltatlist[d],
                                               save = False)
                        c = walk.converging
                        #c = 1
                        #('run','N','max_iter','ystar','deltat') 
                        convergence.append((i, Nlist[j], max_iter, ystarlist[y], deltatlist[d],c))
                        print "converged "+str(c)
        except Exception:
            continue
        pickle.dump(convergence, open("examples/results/results_"+str(i)+ "_seed_"+str(seed)+"_" +str(datetime.datetime.now().date()) + ".pickle","w"))
    print convergence

