#!/usr/local/bin/python3

from sompy3 import sompy3
import numpy as np
from sklearn.cluster import KMeans
import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import matplotlib.cbook
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

from datetime import datetime, date, timedelta
#import subprocess

#--------------------------------------------------------------------------
def bitfield(n,dim):
#--------------------------------------------------------------------------
    a = [1 if digit=='1' else 0 for digit in bin(n)[2:]]
    b = [0 for i in range(dim-len(a))]
    b.extend(a)
    return b

#--------------------------------------------------------------------------
def generate_Data(dim,dlen,shuffle=False,maxclusters=120):
#--------------------------------------------------------------------------
    np.random.seed(0)

    p = int(pow(2,dim))
    nclusters = min(maxclusters,p)

    Data = [None for i in range(nclusters)]
    for k in range(nclusters):
        # bits = np.random.randint(2, size=dim)
        bits = bitfield(k,dim)
        Data[k] = np.random.rand(dlen,dim)
        for d in range(dim):
            Data[k][:,d] = bits[d] + Data[k][:,d]
    Data = np.concatenate(Data)
    if shuffle:
        np.random.shuffle(Data)
    return Data, nclusters



#==========================================================================
#= MAIN ===================================================================
#==========================================================================

# ------------- start timer
compStart = datetime.now()

# ----------- parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--compute", "-c", action="store_true",  dest="compute", default=False, help="compute")
args = parser.parse_args()

dlen        = 100
dim         = 3
size        = 100
DataComplete, nclusters = generate_Data(dim,dlen,shuffle=True)
mapsize     = [size,size]
nbatches    = 4
maxtrainlen = 1
radius      = None

print('='*60)
print('dimension:               {:d}'.format(dim))
print('number of clusters:      {:d}'.format(nclusters))
print('total number of samples: {:d}'.format(len(DataComplete)))
print('mapsize:                 {:d}x{:d}'.format(mapsize[0],mapsize[1]))
print('nbatches:                {:d}'.format(nbatches))
print('maxtrainlen:             {:d}'.format(maxtrainlen))
print('radius:                  {:s}'.format('None' if radius is None else str(radius)))
print('='*60)

if (args.compute):
    m = []
    som = sompy3.som(normalizer='var', mapsize=mapsize, radius=radius)
    for bat in range(nbatches):
        frombatch = int(bat*len(DataComplete)/nbatches)
        tobatch   = int(np.min([(bat+1)*int(len(DataComplete)/nbatches),len(DataComplete)]))
        # print(frombatch,tobatch)
        a = datetime.now()
        batch = DataComplete[frombatch:tobatch]
        som.train(batch,maxtrainlen=maxtrainlen)
        m.append( (datetime.now() - a).total_seconds() )
        print('up to now: computing time for one batch took between {:f} and {:f} s, mean {:f} s'.format(np.min(m),np.max(m),np.mean(m)))
        som.visualizeCodebook(path='./images/',filenameAdd=bat,dimnames=None)
        som.visualizeClusters(nclusters,filenameAdd=bat,text=True,interiorPoints=3,path="./images/")

# ------------- stop timer
compEnd = datetime.now()
sys.stdout.write('Elapsed computing time: '+str(compEnd-compStart)+'\n')
