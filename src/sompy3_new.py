import os
import sys
import numpy as np
import ray
#from multiprocessing.dummy import Pool
#from multiprocessing import cpu_count
#import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import matplotlib.cbook
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
#import threading

import sompy3.normalization.normalization as norm
import sompy3.neighbourhood.neighbourhood as ngb
from scipy.sparse import csr_matrix

EPS = .0000000000000000001

class NotImplementedError(Exception):
    pass

class som(object):

    # ----------------------------------------------------------
    def __init__(self, dim=None, mapsize=None, normalizer='var', neighbourhoodMethod='gaussian', initialization='random', layout='rect'):
    # ----------------------------------------------------------
        if normalizer is not None:
            self.normalizer = norm.compNorm.init(normalizer)
        else:
            self.normalizer = None

        self.mapsize             = [5,5] if mapsize is None else mapsize
        self.nnodes              = self.mapsize[0] * self.mapsize[1]
        self.initialization      = initialization
        self.CodebookInitialized = False
        self.dim                 = dim
        self.layout              = layout

        if self.layout != 'rect':
            raise NotImplementedError("layout {:s} is not yet implemented".format(layout))

        if neighbourhoodMethod is not None:
            self.compNeighbourhood = ngb.compNeighbourhood.init(neighbourhoodMethod,self.mapsize)
        else:
            self.compNeighbourhood = None

        if self.dim is not None:
            self._initializeCodebook()

        ray.init()

    # ========================================================================================================================================
    def _RowColFromNodeIndex(self,nn):
    # ========================================================================================================================================
        return _RowColFromNodeIndex(nn,self.mapsize[0],self.mapsize[1])

    # ========================================================================================================================================
    def _NodeIndexFromRowCol(self,r,c):
    # ========================================================================================================================================
        return _NodeIndexFromRowCol(r,c,self.mapsize[0],self.mapsize[1])

    # ========================================================================================================================================
    def _initializeCodebook(self):
    # ========================================================================================================================================
        if self.initialization == 'random':
            np.random.seed(0)
            # self.codebook = 2*np.random.rand(self.nnodes,self.dim) - np.ones(self.nnodes*self.dim).reshape(self.nnodes,self.dim)
            self.codebook = np.zeros(self.nnodes*self.dim).reshape(self.nnodes,self.dim)
        else:
            raise NotImplementedError("{:s} initialization is not yet implemented".format(initialization))
        self.CodebookInitialized = True

    # ========================================================================================================================================
    def _getTRR(self,radiusIni,radiusFinal,trainlen,maxtrainlen):
    # ========================================================================================================================================
        if radiusIni is None:
            radiusIni   = np.min(self.mapsize)*.1
        if radiusFinal is None:
            radiusFinal = np.min(self.mapsize)*.05
        learningRateIni   = 1
        learningRateFinal = .5
        if trainlen is None:
            trainlen = int(np.min([10,maxtrainlen]))
        return radiusIni,radiusFinal,learningRateIni,learningRateFinal,trainlen


    # ========================================================================================================================================
    def train(self,data,i,trainlen=None,maxtrainlen=np.Inf,radiusIni=None,radiusFinal=None):
    # ========================================================================================================================================
        d = data.shape[1]
        if self.dim is None:
            self.dim = d
        elif self.dim != d:
            raise Exception ("Data dimension mismatch.")

        if not self.CodebookInitialized:
            self._initializeCodebook()

        if self.normalizer is not None:
            print(" Normalizing data...")
            self.data = self.normalizer.normalize(data)
        else:
            self.data = data

        radiusIni,radiusFinal,learningRateIni,learningRateFinal,trainlen = self._getTRR(radiusIni,radiusFinal,trainlen,maxtrainlen)
        radius = np.linspace(radiusIni, radiusFinal, trainlen)
        learningRate = np.linspace(learningRateIni, learningRateFinal, trainlen)
        print('  trainlen {:d} radiusIni: {:f} radiusFinal: {:f}'.format(trainlen,radiusIni,radiusFinal))
        jobs = []
        for step in range(trainlen):
            print('  training step {:d} radius: {:f} learningRate: {:f}'.format(step,radius[step],learningRate[step]))
            for d in self.data[:4]:
                # jobs.append(_train.remote(d,radius[step],learningRate[step],self.codebook,self.mapsize,self.nnodes))
                cookbookUp = _train(d,radius[step],learningRate[step],self.codebook,self.mapsize,self.nnodes)
            #for job in jobs:
                # self.codebook += ray.get(job)
                self.codebook += cookbookUp
            self.visualizeCodebook('./images/test_'+str(i)+'_'+str(step)+'.png')

        return

    # ========================================================================================================================================
    def visualizeCodebook(self,filename):
    # ========================================================================================================================================
        nrows = int(np.ceil(np.sqrt(self.dim)))
        ncols = int(np.ceil(np.sqrt(self.dim)))
        if (ncols-1)*nrows >= self.dim:
            nrows -= 1
        plt.figure(figsize=(nrows,ncols))
        for ind in range(self.dim):
            fig = plt.subplot(nrows, ncols, ind+1, aspect=float(self.mapsize[0]/self.mapsize[1]))
            mp = self.codebook[:, ind].reshape(self.mapsize[0], self.mapsize[1])
            minEntry = np.min(mp)
            maxEntry = np.max(mp)
            norm = pltcol.Normalize(vmin=minEntry, vmax=maxEntry)
            pl = plt.pcolor(mp, norm=norm) #,cmap=plt.cm.get_cmap('RdYlBu_r'))
            plt.axis([0, self.mapsize[1], 0, self.mapsize[0]])
            plt.title('dim '+str(ind),size=10)
            #plt.colorbar(pl)
            ax = plt.gca()
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        plt.tight_layout()
        #plt.axes().set_aspect('equal', 'datalim')
        plt.savefig(filename, dpi = 300)
        return





# @ray.remote
# ========================================================================================================================================
def _train(d,radius,learningRate,codebook,mapsize,nnodes):
# ========================================================================================================================================
    D     = np.array([d,]*nnodes) - codebook
    N     = np.einsum('ij,ij->i', D, D)
    bmu   = np.argmin(N)
    neigh = _neighbourhood(radius,bmu,mapsize).reshape(nnodes,1)
    return learningRate * (neigh*D)


# ========================================================================================================================================
def _neighbourhood(radius,nn,mapsize):
# ========================================================================================================================================
    x,y = _RowColFromNodeIndex(nn,mapsize[0],mapsize[1])
    D = np.zeros(mapsize)
    for i in range(mapsize[0]):
        for j in range(mapsize[1]):
            D[i,j] = (i-x)**2 + (j-y)**2
    return np.exp(-1.0*D/(2*radius**2))

# ========================================================================================================================================
def _NodeIndexFromRowCol(r,c,rowsize,colsize):
# ========================================================================================================================================
    if r >= rowsize or r < 0:
        raise Exception ('bad row index')
    elif c >= colsize or c < 0:
        raise Exception ('bad col index')
    return r*colsize + c

# ========================================================================================================================================
def _RowColFromNodeIndex(nn,rowsize,colsize):
# ========================================================================================================================================
    if nn >= rowsize*colsize or nn < 0:
        raise Exception ('bad node index')
    r = int(np.floor(nn/rowsize))
    c = int(nn%colsize)
    return (r,c)
