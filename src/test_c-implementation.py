#!/usr/local/bin/python3

#from multiprocessing import Process
#import multiprocessing as mp
# from multiprocessing import Process
from multiprocessing import Pool, TimeoutError
import os
import time
import random
import sys
from datetime import datetime
import numpy as np
from sompy3 import sompy3

import ctypes as ct
_trainlib = ct.CDLL('train.so')
_trainlib.train.argtypes = [ct.c_double, ct.POINTER(ct.c_double), ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_int), ct.c_double]

def train(learningRate,d,dim,dlen,codebook,mapsize,radius):
    global _trainlib
    _trainlib.train(learningRate,d.ctypes.data_as(ct.POINTER(ct.c_double)),dim,dlen,codebook.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*2)(*mapsize),radius)
    return



mapsize = [4,4]
nnodes  = mapsize[0]*mapsize[1]
dim     = 10
dlen    = 1
learningRate = .5
radius = np.min(mapsize)*.1

print('making data of size {:d}'.format(dlen*dim))
#data = np.random.rand(1,dlen*dim)
data = np.array([i for i in range(dlen*dim)],dtype=float)
data = data.reshape(dlen,dim)

# print('making codebook of size {:d}'.format(nnodes*dim))
# np.random.seed(0)
# codebook = 2*np.random.rand(nnodes,dim) - np.ones(nnodes*dim).reshape(nnodes,dim)
# codebook_save = codebook

# print('processing data sequentially')
# compStart = datetime.now()

som = sompy3.som(normalizer='var', mapsize=mapsize, radius=radius)
codebook_save = som.codebook

som.train(trainlen=1,maxtrainlen=1)
print(som.codebook)

train(learningRate,data,dim,dlen,codebook_save,mapsize,radius)
print(codebook_save)

# compEnd = datetime.now()
# sys.stdout.write('Elapsed time: '+str(compEnd-compStart)+'\n')







# print('making xpos, ypos')
# xpos = []
# ypos = []
# for nn in range(nnodes):
#     x     = int(nn/mapsize[0])
#     xs    = mapsize[0]   - x
#     xe    = mapsize[0]   + xs
#     xpos.append([xs,xe])
#     y     = int(nn%mapsize[0])
#     ys    = mapsize[1]   - y
#     ye    = mapsize[1]   + ys
#     ypos.append([ys,ye])
#
# print('making neighbourhood')
# radius = np.min(mapsize)*.1
# neighbourhoodBase = np.zeros([3*mapsize[0],3*mapsize[1]])
# for i in range(3*mapsize[0]):
#     for j in range(3*mapsize[1]):
#         neighbourhoodBase[i,j] = (i-mapsize[0])**2 + (j-mapsize[1])**2
# neighbourhoodBase = np.exp(-1.0*neighbourhoodBase/(2*radius**(1.5)))
#
# print('processing data sequentially')
# for i in range(5):
#     compStart = datetime.now()
#
#     # for d in data:
#     #     train(learningRate,d,nnodes,codebook,xpos,ypos)
#
#     compEnd = datetime.now()
#     sys.stdout.write('Elapsed time: '+str(compEnd-compStart)+'\n')
