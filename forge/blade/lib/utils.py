import itertools
import time

import numpy as np
from collections import defaultdict

class staticproperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)

#https://stackoverflow.com/questions/26496831/how-to-convert-defaultdict-of-defaultdicts-of-defaultdicts-to-dict-of-dicts-o
def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

def cosSim(x):
   from sklearn.metrics.pairwise import euclidean_distances as pdist
   mag = np.sqrt(np.sum(x**2, 1))
   x = x / mag.reshape(-1, 1)
   dists = pdist(x)
   return dists

def vstack(x):
   if len(x) > 0:
      return np.vstack(x)
   return []

def groupby(items, key):
   targs = sorted(items, key=key)
   return itertools.groupby(targs, key=key)

#Generic
def uniqueKey(refDict):
   idx = seed()
   if idx in refDict:
      #print('Hash collision--highly unlikely, check your code')
      return uniqueKey(refDict)
   return idx

def seed():
   return int(np.random.randint(0, 2**32))

def invertDict(x):
   return {v: k for k, v in x.items()}

def loadDict(fName):
   with open(fName) as f:
      s = eval(f.read())
   return s

def terminalClasses(cls):
   ret = []
   subclasses = cls.__subclasses__()
   if len(subclasses) == 0:
      ret += [cls]
   else:
      for e in subclasses:
         ret += terminalClasses(e)
   return ret

def l1(pos1, pos2):
   r1, c1 = pos1
   r2, c2 = pos2
   return abs(r1 - r2) + abs(c1 - c2)

def l2(pos1, pos2):
   r1, c1 = pos1
   r2, c2 = pos2
   return np.sqrt((r1 - r2)**2 + (c1 - c2)**2)

def linf(pos1, pos2):
   r1, c1 = pos1
   r2, c2 = pos2
   return max(abs(r1 - r2), abs(c1 - c2))

def norm(x, n=2):
   return (np.sum(np.abs(x)**n)**(1.0/n)) / np.prod(x.shape)

#Bounds checker
def inBounds(r, c, shape, border=0):
   R, C = shape
   return (
         r > border and
         c > border and
         r < R - border and
         c < C - border
         )

#Because the numpy version is horrible
def randomChoice(aList):
   lLen = len(aList)
   ind = np.random.randint(0, lLen)
   return aList[ind]

#Tracks inds of a permutation
class Perm():
   def __init__(self, n):
      self.inds = np.random.permutation(np.arange(n))
      self.m = n
      self.pos = 0

   def next(self, n):
      assert(self.pos + n < self.m)
      ret = self.inds[self.pos:(self.pos+n)]
      self.pos += n
      return ret

#Exponentially decaying average
class EDA():
   def __init__(self, k=0.9):
      self.k = k 
      self.eda = None
   
   def update(self, x):
      if self.eda is None:
         self.eda = x
         return
      #self.eda = self.eda * k / (x * (1-k))
      self.eda = (1-self.k)*x + self.k*self.eda

class Timer:
   def __init__(self):
      self.start = time.time()

   def ticked(self, delta):
      if time.time() - self.start > delta:
         self.start = time.time()
         return True
      return False

class BenchmarkTimer:
   def __init__(self):
      self.eda = EDA() 
      self.accum = 0
   def startRecord(self):
      self.start = time.time()

   def stopRecord(self, accum=False):
      if accum:
         self.accum += time.time() - self.start
      else:
         self.eda.update(self.accum + time.time() - self.start)
         self.accum = 0

   def benchmark(self):
      return self.eda.eda
      
#Continuous moving average
class CMA():
   def __init__(self):
      self.t = 1.0
      self.cma = None

   def update(self, x):
      if self.cma is None:
         self.cma = x
         return
      self.cma = (x + self.t*self.cma)/(self.t+1)
      self.t += 1.0

#Continuous moving average
class CMV():
   def __init__(self):
      self.cma = CMA()
      self.cmv = None

   def update(self, x):
      if self.cmv is None:
         self.cma.update(x)
         self.cmv = 0
         return
      prevMean = self.cma.cma
      self.cma.update(x)
      self.cmv += (x-prevMean)*(x-self.cma.cma)

   @property
   def stats(self):
      return self.cma.cma, self.cmv

def matCrop(mat, pos, stimSz):
   ret = np.zeros((2*stimSz+1, 2*stimSz+1), dtype=np.int32)
   R, C = pos
   rt, rb = R-stimSz, R+stimSz+1
   cl, cr = C-stimSz, C+stimSz+1
   for r in range(rt, rb):
      for c in range(cl, cr):
         if inBounds(r, c, mat.shape):
            ret[r-rt, c-cl] = mat[r, c]
         else:
            ret[r-rt, c-cl] = 0
   return ret

