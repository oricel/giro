import pandas as pd
import numpy as np

import functools
import time
import math

import networkx as nx
import matplotlib.pyplot as plt
import collections

import random 
from networkx.utils import powerlaw_sequence

import pickle
import datetime

def test_aw():
    print "aw"
    
def pheno(B, X):
    return np.dot(B,np.transpose(X))[0,0]
    
def mismatch(y, ystar, M0=2, mu=0.01, epsilon=3):
    m = (M0/2.0) * (1 + np.tanh((abs(y-ystar)-epsilon)/mu))
    return m

def converged(values, value=0, ratio=0.2):
    n = len(values)
    m = int((1-ratio)*n)
    return all([ v == value for v in values[m:n]])
    
class Cell(object):
    def __init__(self, N, alpha=1.0, beta=2.4,
                 c=0.5, alph=100,
                 D=0.001,
                 g0=10
                ):
        # Network -- adjecency matrix
        # N -- number of nodes
        self.N = N
        self.alpha=alpha
        self.beta=beta
        # NK -- number of edges
        # K -- average degree

        # c -- proportion of nonzero b
        self.c = c
        # alph -- parameter for instantiating b
        self.alph = alph
        # g0 -- parameter for instantiating b, J, x
        self.g0 = g0
        # D -- parameter for instantiating J
        self.D = D
        
        # J0 -- initial network strengths
        # J
        # X0 -- initial xi 
        # B -- initial bi
        
        self.init_network()
        self.init_pheno()
        
    def init_network(self):
        N = self.N
        #z=nx.utils.create_degree_sequence(N,powerlaw_sequence)
        while True:
            def seq(n):
                return [random.gammavariate(alpha=self.alpha,beta=self.beta) for i in range(N)]   
            z_in=nx.utils.create_degree_sequence(N,seq)
            def seq(n):
                return [random.gammavariate(alpha=self.alpha,beta=self.beta) for i in range(N)]    
            z_out=nx.utils.create_degree_sequence(N,seq)
            if (sum(z_in) == sum(z_out)):
                break
        nx.is_valid_degree_sequence(z_in)
        nx.is_valid_degree_sequence(z_out)

        G=nx.directed_configuration_model(z_in,z_out)  # configuration model

        # remove multiple edges
        G=nx.DiGraph(G)
        # remove self loops
        G.remove_edges_from(G.selfloop_edges())

        NK = len(G.edges())
        
        self.Network = nx.to_numpy_matrix(G, nodelist=range(N))
        self.NK = NK
        self.K = NK/(N*1.)
        return None

    def init_pheno(self):
        c = self.c
        alph = self.alph
        D = self.D
        g0 = self.g0
        N = self.N
        K = self.K
        self.J0 = np.matrix(np.random.normal(0, np.sqrt((g0**2)/K), N*N).reshape(N,N))
        
        # Instantiate b (Nx1)
        # number of nonzero b elements
        nonzerob = int(c*N)
        # bi that will be nonzero
        nonzeroN = np.random.choice(N, size=nonzerob, replace=False, p=None)
        # B indices that will be set to 0
        BZero = list(set(range(0,N)) - set(nonzeroN))
        B = np.random.normal(0, np.sqrt(alph/(g0**2*c*N)), N)
        B[BZero] = 0
        B = np.asmatrix(B)
        X = np.asmatrix(np.random.normal(0,g0,N))
        
        self.B = B
        self.X0 = X
        
        return None
                
class Walk(object):
    def __init__(self, cell, max_iter, ystar, M0=2, mu=0.01, epsilon=3, deltat=5, ratio=0.2):
        self.J=cell.J0
        J=self.J
        self.B=cell.B
        B=self.B
        self.T=cell.Network
        T=self.T
        self.X=cell.X0
        X=self.X
        self.D=cell.D
        D=self.D
        self.N=cell.N
        N=self.N
        self.ystar=ystar
        self.max_iter=max_iter
        self.M0=M0
        self.mu=mu
        self.epsilon=epsilon
        self.deltat=deltat
        self.ratio=ratio
        
        ylist = [pheno(B,X)]
        xlist = [X]
        jlist = [J]
        xdeltalist = [X]
        xreslist = [X]
        wlist = [np.multiply(T, J)]
        mlist = [mismatch(pheno(B,X), ystar=ystar, M0=M0, mu=mu, epsilon=epsilon)]
        
        vtanh=np.vectorize(np.tanh)
            
        i = 0
        while (i < max_iter):
            # update J
            m = mismatch(pheno(B,X), ystar=ystar, M0=M0, mu=mu, epsilon=epsilon)
            #print "iter "+str(i)+", mismatch "+str(m)
            deltaJ = np.multiply(math.sqrt(D * m), 
                                 np.random.normal(0, 1, N*N).reshape(N,N))
            J = np.add(J, deltaJ)
            # update X
            W = np.multiply(T, J)
            # compute saturation function Fi
            Fi = vtanh(X)
            
            Xdelta = np.multiply(1./deltat,np.transpose(np.dot(W,np.multiply(1.,np.transpose(Fi)))))
            Xres = np.multiply(1.-1./deltat, X)
            X = np.add(Xdelta,Xres)
            
            # iterate
            i += 1
            ylist.append(pheno(B,X))
            xlist.append(X)
            jlist.append(J)
            wlist.append(W)
            xreslist.append(Xres)
            xdeltalist.append(Xdelta)
            mlist.append(mismatch(pheno(B,X), ystar=ystar, M0=M0, mu=mu, epsilon=epsilon))
            
        self.ylist=ylist
        self.xlist=xlist
        self.jlist=jlist
        self.wlist=wlist
        self.xreslist=xreslist
        self.xdeltalist=xdeltalist
        self.mlist=mlist
        self.converging=converged(mlist, ratio=ratio)
    
    def plot(self, n=30, filename=None):
        max_iter = self.max_iter
        indx = np.where(self.T>0)
        xi,yi = indx
        xb, yb = np.where(self.B>0)
        
        #print xi, yi
        #print xb, yb
        #print len(self.jlist)
        #print self.jlist[1]
        #print len(self.wlist)
        #print self.wlist[1]
        #print len(self.xlist)
        #print self.xlist[1]

        jselect = []
        wselect = []
        xselect = []
        xresselect = []
        xdeltaselect = []
        mselect = []

        n = min(n, len(xb))
        for j in range(0,n):
            jselect.append([self.jlist[i][xi[j],yi[j]] for i in range(0,max_iter)])
            wselect.append([self.wlist[i][xi[j],yi[j]] for i in range(0,max_iter)])
            xselect.append([self.xlist[i][xb[j],yb[j]] for i in range(0,max_iter)])
            xresselect.append([self.xreslist[i][xb[j],yb[j]] for i in range(0,max_iter)])
            xdeltaselect.append([self.xdeltalist[i][xb[j],yb[j]] for i in range(0,max_iter)])

        fig = plt.figure(figsize=(14,8))
        
        plt1 = fig.add_subplot(4,2,1)
        plt1.plot(self.ylist)
        plt1.plot([self.ystar for k in range(0,max_iter+1)])
        plt1.set_ylabel('y')

        plt2 = fig.add_subplot(4,2,2)
        for k in range(0,len(jselect)):
            plt2.plot(jselect[k])
            plt2.set_ylabel('J')

        plt3 = fig.add_subplot(4,2,3)
        for k in range(0,len(xselect)):
            plt3.plot(xselect[k],linewidth=0.5)
            plt3.set_ylabel('X')

        plt4 = fig.add_subplot(4,2,4)
        for k in range(0,len(wselect)):
            plt4.plot(wselect[k],linewidth=0.5)
            plt4.set_ylabel('W')

        plt5 = fig.add_subplot(4,2,5)
        for k in range(0,len(xdeltaselect)):
            plt5.plot(xdeltaselect[k],linewidth=0.5)
            plt5.set_ylabel('Delta X')

        plt6 = fig.add_subplot(4,2,6)
        for k in range(0,len(xresselect)):
            plt6.plot(xresselect[k],linewidth=0.5)
            plt6.set_ylabel('X Residue')

        plt7 = fig.add_subplot(4,2,7)
        plt7.plot(self.mlist)
        plt7.set_ylabel('Mismatch')

        if (filename):
            plt.savefig(filename+".png")
        plt.show()

    def save(self, filename=None):
        now = datetime.datetime.now().date()
        pickle.dump(self,
                    open(filename+"nodes."+str(self.N)+"_maxiter."+str(self.max_iter)+
                         "_converged."+str(self.converging)+"_"+str(now)+ ".pickle","w"))
        return
    