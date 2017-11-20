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

from awalker import randomcell


if __name__ == '__main__':
    Nlist = [500,1000,1200,1500]
    max_iter = 2500
    ystarlist = [0,5,10]
    deltatlist = [1,2,3,4,4.5,5,5.5,6,7,9,11,13]

    convergence = {}
    n_runs = 2
    for i in range(0,n_runs):
        try:
            for j in range(0,len(Nlist)):
                for d in range(0,len(deltatlist)):
                    for y in range(0, len(ystarlist)):
                        cell = randomcell.Cell(Nlist[j])    
                        print "run " + str(i) + " with " + str(Nlist[j]) + " nodes, " + str(max_iter) + " max iterations, " + str(ystarlist[y]) + " ystar, " + str(deltatlist[d]) + " deltat"
                        walk = randomcell.Walk(cell, max_iter=max_iter,
                                               ystar = ystarlist[y],
                                               deltat = deltatlist[d],
                                               save = False)
                        c = walk.converging
                        convergence['run','N','max_iter','ystar','deltat'] = i, Nlist[j], max_iter, ystarlist[y], deltatlist[d]
                        print "converged "+str(c)
        except Exception:
            continue
    print convergence

    pickle.dump(convergence, open("examples/results"+str(datetime.datetime.now().date()) + ".pickle","w"))
