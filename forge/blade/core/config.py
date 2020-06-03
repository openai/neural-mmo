from pdb import set_trace as T
import numpy as np
import inspect
import os

from collections import defaultdict
from itertools import chain

class StaticIterable(type):
   def __iter__(cls):
      stack = list(cls.__dict__.items())
      for name, attr in cls.__dict__.items():
         if name == '__module__':
            continue
         if name.startswith('__'):
            break
         yield name, attr

class Template(metaclass=StaticIterable):
   def __init__(self):
      self.data = {}
      cls       = type(self)

      #Set defaults from static properties
      for k, v in cls:
         self.set(k, v)

   def override(self, **kwargs):
      for k, v in kwargs.items():
         err = 'CLI argument: {} is not a Config property'.format(k)
         assert hasattr(self, k), err
         self.set(k, v)

   def set(self, k, v):
      setattr(self, k, v)
      self.data[k] = v

   def print(self):
      keyLen = 0
      for k in self.data.keys():
         keyLen = max(keyLen, len(k))

      print('Configuration')
      for k, v in self.data.items():
         print('   {:{}s}: {}'.format(k, keyLen, v))

class Config(Template):
   '''An environment configuration object'''
   ROOT = os.path.join(os.getcwd(), 'resource/maps/procedural/map')
   SUFFIX = '/map.tmx'

   NAME_PREFIX = 'Neural_'

   SZ = 128

   SZ = 62
   BORDER = 9
   R = C = SZ + BORDER

   STIM = 7
   WINDOW = 2*STIM + 1

   NENT = 256
   NPOP = 8
   NTILE = 6 #Add this to tile static

   RESOURCE = 10
   HEALTH   = 10

   RESOURCERESTORE = 0.5
   HEALTHRESTORE   = 0.1

   XPSCALE = 10
   IMMUNE  = 10

   #Attack ranges
   MELEERANGE = 1
   RANGERANGE = 2
   MAGERANGE  = 3

   def SPAWN(self):
      R, C = Config.R, Config.C
      spawn, border, sz = [], Config.BORDER, Config.SZ
      spawn += [(border, border+i) for i in range(sz)]
      spawn += [(border+i, border) for i in range(sz)]
      spawn += [(R-1, border+i) for i in range(sz)]
      spawn += [(border+i, C-1) for i in range(sz)]
      idx = np.random.randint(0, len(spawn))
      return spawn[idx]
