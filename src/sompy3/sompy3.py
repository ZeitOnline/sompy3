
import os
import sys
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import matplotlib.cbook
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import glob
from collections import Counter

import sompy3.normalization.normalization as norm
from scipy.sparse import csr_matrix

import ctypes as ct
#_trainlib = ct.CDLL('./train.so')
trainfile = ''
filelist = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','build*','lib.*','*.so'))
if 'darwin' in sys.platform.lower():
    platform = 'darwin'
elif 'linux' in sys.platform.lower():
    platform = 'linux'
elif 'win' in sys.platform.lower():
    platform = 'win'
for file in filelist:
    if 'train' in file and file.endswith('.so') and platform in file:
        trainfile = file
_trainlib = ct.CDLL(trainfile, ct.RTLD_GLOBAL)
_trainlib.trainSequential.argtypes = [ct.c_double, ct.POINTER(ct.c_double), ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_int), ct.c_double]
_trainlib.trainParallel.argtypes = [ct.c_double, ct.POINTER(ct.c_double), ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_int), ct.c_double, ct.c_int]
_trainlib.getBMUlistSequential.argtypes = [ct.POINTER(ct.c_double), ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_int), ct.POINTER(ct.c_long)]
_trainlib.getBMUlistParallel.argtypes = [ct.POINTER(ct.c_double), ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_int), ct.POINTER(ct.c_long), ct.c_int]
_trainlib.getUMatrix.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_int), ct.c_int, ct.POINTER(ct.c_double)]

def trainSequential(learningRate,d,dim,dlen,codebook,mapsize,radius):
    global _trainlib
    _trainlib.trainSequential(learningRate,d.ctypes.data_as(ct.POINTER(ct.c_double)),dim,dlen,codebook.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*2)(*mapsize),radius)
    return

def trainParallel(learningRate,d,dim,dlen,codebook,mapsize,radius,ncores=4):
    global _trainlib
    _trainlib.trainParallel(learningRate,d.ctypes.data_as(ct.POINTER(ct.c_double)),dim,dlen,codebook.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*2)(*mapsize),radius,ncores)
    return

def getBMUlistSequential(data,dim,dlen,codebook,mapsize,bmuList):
    global _trainlib
    _trainlib.getBMUlistSequential(data.ctypes.data_as(ct.POINTER(ct.c_double)),dim,dlen,codebook.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*2)(*mapsize),bmuList.ctypes.data_as(ct.POINTER(ct.c_long)))
    return

def getBMUlistParallel(data,dim,dlen,codebook,mapsize,bmuList,ncores=4):
    global _trainlib
    _trainlib.getBMUlistParallel(data.ctypes.data_as(ct.POINTER(ct.c_double)),dim,dlen,codebook.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*2)(*mapsize),bmuList.ctypes.data_as(ct.POINTER(ct.c_long)),ncores)
    return

def getUMatrix(codebook,mapsize,dim,umatrix):
    global _trainlib
    _trainlib.getUMatrix(codebook.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*2)(*mapsize),dim,umatrix.ctypes.data_as(ct.POINTER(ct.c_double)))
    return

class NotImplementedError(Exception):
    pass

class som(object):

    # ----------------------------------------------------------
    def __init__(self, dim=None, mapsize=None, normalizer='var', initialization='random', layout='rect', radius=None):
    # ----------------------------------------------------------
        if normalizer is not None:
            self.normalizer = norm.compNorm.init(normalizer)
        else:
            self.normalizer = None

        self.mapsize         = [5,5] if mapsize is None else mapsize
        self.nnodes          = self.mapsize[0] * self.mapsize[1]
        self.initialization  = initialization
        self.initialized     = False
        self.dim             = dim
        self.layout          = layout
        self.umatrix         = None
        self.clusters        = None
        self.nclusters       = None

        if self.layout != 'rect':
            raise NotImplementedError("layout {:s} is not yet implemented".format(layout))

        if self.dim is not None:
            self._initializeCodebook()

        self.radius = np.min(mapsize)*.1 if radius is None else radius

    # ========================================================================================================================================
    def _initializeCodebook(self):
    # ========================================================================================================================================
        if self.initialization == 'random':
            np.random.seed(0)
            self.codebook = 2*np.random.rand(self.nnodes,self.dim) - np.ones(self.nnodes*self.dim).reshape(self.nnodes,self.dim)
            # self.codebook = np.zeros(self.nnodes*self.dim).reshape(self.nnodes,self.dim)

            self.xpos = []
            self.ypos = []
            for nn in range(self.nnodes):
                x     = int(nn/self.mapsize[0])
                xs    = self.mapsize[0]   - x
                xe    = self.mapsize[0]   + xs
                self.xpos.append([xs,xe])
                y     = int(nn%self.mapsize[0])
                ys    = self.mapsize[1]   - y
                ye    = self.mapsize[1]   + ys
                self.ypos.append([ys,ye])

            self.initialized = True

        else:
            raise NotImplementedError("{:s} initialization is not yet implemented".format(initialization))

    # ========================================================================================================================================
    def _getTRTL(self,trainlen,maxtrainlen):
    # ========================================================================================================================================
        learningRateIni   = .5
        learningRateFinal = .2
        if trainlen is None:
            trainlen = int(np.min([10,maxtrainlen]))
        return learningRateIni,learningRateFinal,trainlen


    # ========================================================================================================================================
    def train(self,data,trainlen=None,maxtrainlen=np.Inf,parallel=False):
    # ========================================================================================================================================
        d = data.shape[1]
        if self.dim is None:
            self.dim = d
        elif self.dim != d:
            raise Exception ("Data dimension mismatch. Dim is {:d}, data dim is {:d}.".format(self.dim,d))
        self.dlen = data.shape[0]

        if not self.initialized:
            self._initializeCodebook()

        if self.normalizer is not None:
            print("Normalizing data...")
            data = self.normalizer.normalize(data)
        else:
            data = data/np.max(np.absolute(data))

        learningRateIni,learningRateFinal,trainlen = self._getTRTL(trainlen,maxtrainlen)
        learningRate = np.linspace(learningRateIni, learningRateFinal, trainlen)
        radius = np.linspace(.5*min(self.mapsize), .1*min(self.mapsize), trainlen)

        print(' total: {:d} training steps, {:d} samples'.format(trainlen,data.shape[0]))

        for step in range(trainlen):
            if (parallel):
                print('   training step {:d}, learningRate: {:f}, radius: {:f}, parallel'.format(step,learningRate[step],radius[step]))
            else:
                print('   training step {:d}, learningRate: {:f}, radius: {:f}, sequential'.format(step,learningRate[step],radius[step]))

            if (parallel):
                trainParallel(learningRate[step],data,self.dim,self.dlen,self.codebook,self.mapsize,radius[step])
            else:
                trainSequential(learningRate[step],data,self.dim,self.dlen,self.codebook,self.mapsize,radius[step])


        return

    # ========================================================================================================================================
    def _NodeIndexFromRowCol(self,r,c):
    # ========================================================================================================================================
        rowsize = self.mapsize[0]
        colsize = self.mapsize[1]
        if r >= rowsize or r < 0:
            raise Exception ('bad row index')
        elif c >= colsize or c < 0:
            raise Exception ('bad col index')
        return r*colsize + c

    # ========================================================================================================================================
    def _RowColFromNodeIndex(self,nn):
    # ========================================================================================================================================
        rowsize = self.mapsize[0]
        colsize = self.mapsize[1]
        if nn >= rowsize*colsize or nn < 0:
            raise Exception ('bad node index')
        r = int(np.floor(nn/rowsize))
        c = int(nn%colsize)
        return (r,c)


    # ========================================================================================================================================
    def visualizeCodebook(self,path='',filenameAdd=None,dimnames=None,filePrefix=None):
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
            if dimnames is None:
                plt.title('dim '+str(ind),size=10)
            else:
                plt.title(dimnames[ind],size=7)
            #plt.colorbar(pl)
            ax = plt.gca()
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        plt.tight_layout()
        #plt.axes().set_aspect('equal', 'datalim')

        if filenameAdd is not None:
            fn = 'codebook_'+str(filenameAdd)+'.png'
        else:
            fn = 'codebook.png'
        if filePrefix is not None:
            fn = str(filePrefix)+fn
        filename = os.path.join(path,fn)
        plt.savefig(filename, dpi = 300)
        plt.close()
        return

    #--------------------------------------------------------------------------
    def visualizeClusters(self,kmeansClusters=None,text=False,path='',filenameAdd=None,filePrefix=None,centered=False,update=True):
    #--------------------------------------------------------------------------

        if (update):
            self._updateUmatrix()

        # save clusters
        self.computeClusters(kmeansClusters=kmeansClusters)
        print('  nclusters={:d}'.format(self.nclusters))

        minEntry = np.min(self.clusters)
        maxEntry = np.max(self.clusters)
        norm = pltcol.Normalize(vmin=minEntry, vmax=maxEntry)
        plt.figure(figsize=(10,10))
        pl = plt.pcolor(self.clusters, norm=norm, cmap=plt.cm.get_cmap('viridis'))

        if (text):
            if not (centered):
                for i in range(0,self.mapsize[0],3):
                    for j in range(0,self.mapsize[1],3):
                        plt.text(j+.5,i+.5,self.clusters[i,j],horizontalalignment='center',verticalalignment='center',size=10)
            else: # centered
                for cluster in range(self.nclusters):
                    cset = np.array([(row,col) for row in range(self.mapsize[0]) for col in range(self.mapsize[1]) if self.clusters[row,col] == cluster ])
                    # print(cluster)
                    # print(cset)
                    meanpos = np.mean(cset,axis=0)
                    i = int(meanpos[0])
                    j = int(meanpos[1])
                    #print(i,j)
                    plt.text(j+.5,i+.5,cluster,horizontalalignment='center',verticalalignment='center',size=40,color='white')

        plt.tight_layout()
        if filenameAdd is not None:
            fn = 'clusters_'+str(filenameAdd)+'.png'
        else:
            fn = 'clusters.png'
        if filePrefix is not None:
            fn = str(filePrefix)+fn
        filename = os.path.join(path,fn)
        plt.savefig(filename, dpi = 100)
        plt.close()

        # save umatrix
        minEntry = np.min(self.umatrix)
        maxEntry = np.max(self.umatrix)
        norm = pltcol.Normalize(vmin=minEntry, vmax=maxEntry)
        plt.figure(figsize=(10,10))
        pl = plt.pcolor(self.umatrix, norm=norm, cmap=plt.cm.get_cmap('viridis'))
        plt.tight_layout()
        plt.colorbar()
        if filenameAdd is not None:
            fn = 'umatrix_'+str(filenameAdd)+'.png'
        else:
            fn = 'umatrix.png'
        if filePrefix is not None:
            fn = str(filePrefix)+fn
        filename = os.path.join(path,fn)
        plt.savefig(filename, dpi = 100)
        plt.close()

        return

    # ========================================================================================================================================
    def _interior(self,points):
    # ========================================================================================================================================
        n = len(points)
        pxy = sorted(points,key=lambda x: (x[0],x[1]))
        interiorPointsY = []
        for i in range(1,n-1):
            x,y = pxy[i]
            if pxy[i-1][0] == x and pxy[i-1][1] == y-1 and pxy[i+1][0] == x and pxy[i+1][1] == y+1:
                interiorPointsY.append([x,y])
        pyx = sorted(points,key=lambda x: (x[1],x[0]))
        interiorPoints = []
        for i in range(1,n-1):
            x,y = pyx[i]
            if pyx[i-1][0] == x-1 and pyx[i-1][1] == y and pyx[i+1][0] == x+1 and pyx[i+1][1] == y and [x,y] in interiorPointsY:
                interiorPoints.append([x,y])
        return np.array(interiorPoints)

    # ========================================================================================================================================
    def computeClusters(self,update=False,kmeansClusters=None):
    # ========================================================================================================================================

        if kmeansClusters == None:
            print('Clustering... (water)')
        else:
            print('Clustering... (kmeans)')

        if (update) or (self.umatrix is None):
            self._updateUmatrix()

        if (kmeansClusters is not None):
             kmeans = KMeans(n_clusters=kmeansClusters, random_state=0).fit(self.codebook)
             self.clusters = kmeans.labels_.reshape(self.mapsize[0], self.mapsize[1])
        else:
            self.clusters = np.array([np.nan for i in range(self.nnodes)]).reshape(self.mapsize)
            sortedIndices = np.unravel_index(np.argsort(self.umatrix, axis=None), self.mapsize)
            actCluster = 0
            for pos in zip(sortedIndices[0],sortedIndices[1]):
                row,col = pos
                ncl = self._findClustersInNeighbourhood(row,col)
                # if ncl == 6:
                #     print(row,col,ncl)
                if np.isnan(ncl):
                    self.clusters[row,col] = actCluster
                    # print(self.umatrix[row,col],row,col,actCluster)
                    actCluster += 1
                else:
                    self.clusters[row,col] = ncl
                    # print(self.umatrix[row,col],row,col,ncl)

        self.clusters = self.clusters.astype(np.int, copy=False)
        self.nclusters = int(np.max(self.clusters)+1)

        return

    # ========================================================================================================================================
    def _findClustersInNeighbourhood(self,row,col,distBound=3):
    # ========================================================================================================================================
        neighbourClusters = []
        for i in range(row-distBound,row+distBound+1):
            for j in range(col-distBound,col+distBound+1):
                if i >= 0 and i < self.mapsize[0] and j >= 0 and j < self.mapsize[1]:
                    if not np.isnan(self.clusters[i,j]):
                        neighbourClusters.append(self.clusters[i,j])
                else:
                    pass
        if len(neighbourClusters) > 0:
            return Counter(neighbourClusters).most_common()[0][0]
        else:
            return np.nan

    # ========================================================================================================================================
    def _updateUmatrix(self):
    # ========================================================================================================================================
        umatrix = np.zeros(self.mapsize,dtype=float)
        getUMatrix(self.codebook,self.mapsize,self.dim,umatrix)
        self.umatrix = umatrix
        return
