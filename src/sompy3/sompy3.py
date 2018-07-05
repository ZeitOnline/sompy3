
import os
import sys
import numpy as np
# from multiprocessing import Pool, TimeoutError
#from multiprocessing.dummy import Pool
#from multiprocessing import cpu_count
#import itertools
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import matplotlib.cbook
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import glob
#import threading

import sompy3.normalization.normalization as norm
import sompy3.neighbourhood.neighbourhood as ngb
from scipy.sparse import csr_matrix

EPS = .0000000000000000001

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
_trainlib.trainParallel.argtypes = [ct.c_double, ct.POINTER(ct.c_double), ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_int), ct.c_double]

def trainSequential(learningRate,d,dim,dlen,codebook,mapsize,radius):
    global _trainlib
    _trainlib.trainSequential(learningRate,d.ctypes.data_as(ct.POINTER(ct.c_double)),dim,dlen,codebook.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*2)(*mapsize),radius)
    return

def trainParallel(learningRate,d,dim,dlen,codebook,mapsize,radius):
    global _trainlib
    _trainlib.trainParallel(learningRate,d.ctypes.data_as(ct.POINTER(ct.c_double)),dim,dlen,codebook.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*2)(*mapsize),radius)
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

        if self.layout != 'rect':
            raise NotImplementedError("layout {:s} is not yet implemented".format(layout))

        if self.dim is not None:
            self._initializeCodebook()

        self.radius = np.min(mapsize)*.1 if radius is None else radius
        self.neighbourhoodBase = np.zeros([3*self.mapsize[0],3*self.mapsize[1]])
        for i in range(3*self.mapsize[0]):
            for j in range(3*self.mapsize[1]):
                self.neighbourhoodBase[i,j] = (i-self.mapsize[0])**2 + (j-self.mapsize[1])**2
        self.neighbourhoodBase = np.exp(-1.0*self.neighbourhoodBase/(2*self.radius**(1.5)))

        # ray.init()

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

            #for d in data:
                #print(self.codebook + self._train(d,learningRate[step]))
            #trainSequential(learningRate[step],data,self.dim,self.dlen,self.codebook,self.mapsize,self.radius)
            if (parallel):
                trainParallel(learningRate[step],data,self.dim,self.dlen,self.codebook,self.mapsize,radius[step])
            else:
                trainSequential(learningRate[step],data,self.dim,self.dlen,self.codebook,self.mapsize,radius[step])

            # batchsize = 4
            # datalen = data.shape[0]
            # for b in range(0,datalen,batchsize):
            #     print('batch',b,min(datalen,b+batchsize))
            #     with Pool(processes=4) as pool:
            #         jobs = [pool.apply_async(_trainParallel, (data[d],learningRate[step],self.nnodes,self.codebook,self.xpos,self.ypos,self.neighbourhoodBase,)) for d in range(b,min(datalen,b+batchsize))]
            #     print('jobs:',len(jobs))
            #     for res in [job.get() for job in jobs]:
            #         self.codebook += r

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
    def visualizeCodebook(self,path='',filenameAdd=None,dimnames=None):
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
            filename = os.path.join(path,'codebook_'+str(filenameAdd)+'.png')
        else:
            filename = os.path.join(path,filenamePrefix+'.png')
        plt.savefig(filename, dpi = 300)
        plt.close()
        return

    #--------------------------------------------------------------------------
    def visualizeClusters(self,nclusters,text=False,dots=True,interiorPoints=None,path='',filenameAdd=None):
    #--------------------------------------------------------------------------
        kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(self.codebook)
        mp = kmeans.labels_.reshape(self.mapsize[0], self.mapsize[1])
        minEntry = np.min(mp)
        maxEntry = np.max(mp)
        norm = pltcol.Normalize(vmin=minEntry, vmax=maxEntry)
        norm = pltcol.Normalize(vmin=minEntry, vmax=maxEntry)

        plt.figure(figsize=(10,10))
        pl = plt.pcolor(mp, norm=norm, cmap=plt.cm.get_cmap('viridis'))
        if (text):
            for i in range(0,mp.shape[0],3):
                for j in range(0,mp.shape[1],3):
                    plt.text(j+.5,i+.5,mp[i,j],horizontalalignment='center',verticalalignment='center',size=10)
        if (dots):
            for cluster in range(nclusters):
                C = np.array([ self._RowColFromNodeIndex(i) for i in range(self.nnodes) if kmeans.labels_[i] == cluster])
                if interiorPoints is None:
                    c = np.mean(C,axis=0)
                    row,col = int(c[0]),int(c[1])
                    plt.text(col+.5,row+.5,cluster,color='white',size=30)
                else:
                    CO = C
                    for i in range(interiorPoints):
                        C = self._interior(C)
                        if len(C) < 5:
                            C = CO
                            print('WARNING: {:d} interior steps'.format(i-1))
                            break
                        else:
                            CO = C
                    # print('lenC:',len(C))
                    plt.scatter(C[:,1],C[:,0],s=10,c='white')
        plt.tight_layout()
        if filenameAdd is not None:
            filename = os.path.join(path,'clusters_'+str(filenameAdd)+'.png')
        else:
            filename = os.path.join(path,'clusters.png')
        plt.savefig(filename, dpi = 100)
        plt.close()

        nrows = int(np.ceil(np.sqrt(nclusters)))
        ncols = int(np.ceil(np.sqrt(nclusters)))
        if (ncols-1)*nrows >= nclusters:
            nrows -= 1
        plt.figure(figsize=(nrows,ncols))
        x = np.arange(self.dim)
        for cluster in range(nclusters):
            fig = plt.subplot(nrows, ncols, cluster+1)
            C = np.array([ self._RowColFromNodeIndex(i) for i in range(self.nnodes) if kmeans.labels_[i] == cluster])
            if interiorPoints is None:
                c = np.mean(C,axis=0)
                row,col = int(c[0]),int(c[1])
                nn = self._NodeIndexFromRowCol(row,col)
                s = self.codebook[nn]
            else:
                # for i in range(interiorPoints):
                #    C = self._interior(C)
                s = np.mean([self.codebook[self._NodeIndexFromRowCol(row,col)] for (row,col) in C],axis=0)
                #print([self.codebook[self._NodeIndexFromRowCol(row,col)] for (row,col) in C])
                #print(s)
            plt.bar(x, s, .5, align='center')
            plt.ylim(ymin=-1,ymax=1)
            plt.title('cluster ' + str(cluster), size=10)
        plt.tight_layout()
        if filenameAdd is not None:
            filename = os.path.join(path,'cluster_details_'+str(filenameAdd)+'.png')
        else:
            filename = os.path.join(path,'cluster_details.png')
        plt.savefig(filename, dpi = 300)
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
    def _train(self,d,learningRate):
    # ========================================================================================================================================
        # print('py working with this vector:',d)
        # print('py codebook:')
        # print(self.codebook)
        D     = np.array([d,]*self.nnodes) - self.codebook
        E = np.einsum('ij,ij->i', D, D)
        # print('py E:')
        # print(E)
        bmu   = np.argmin(E)
        # print('py bmu:',bmu)
        xs,xe = self.xpos[bmu]
        ys,ye = self.ypos[bmu]
        neigh = self.neighbourhoodBase[xs:xe,ys:ye].reshape(self.nnodes,1)
        # print('py neigh:')
        # print(learningRate*neigh)
        return learningRate * (neigh*D)
