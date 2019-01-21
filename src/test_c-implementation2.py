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

import ctypes as ct
_trainlib = ct.CDLL('train.so')
_trainlib.test.argtypes = [ct.POINTER(ct.c_double), ct.c_int, ct.c_int]

def test(data,rows,cols):
    global _trainlib
    _trainlib.test(data.ctypes.data_as(ct.POINTER(ct.c_double)),rows,cols)
    return



rows = 4
cols = 3
dim = 10
print('making data')
#data = np.random.rand(1,dimx*dimy)
data = np.array([i for i in range(rows*cols*dim)],dtype=float).reshape(rows*cols,dim)
print(data)
d = data.reshape(rows,cols,dim)
print(data[:,0])
print(d[:,:,0])
test(data,rows,cols)
