import pandas as pd
import numpy as np

import functools
import time
import math


import networkx as nx
import matplotlib.pyplot as plt
import collections

import random 
import networkx as nx
from networkx.utils import powerlaw_sequence

from awalker import randomcell


if __name__ == '__main__':
    N = 500
    convergedlist = []
    n_runs = 1000
    for i in range(0,n_runs):
        cell = randomcell.Cell(N)
        walk = randomcell.Walk(cell, max_iter=500, ystar=5)
        convergedlist.append(walk.converging)
    print convergedlist.count(True)