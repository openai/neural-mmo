from pdb import set_trace as T
import numpy as np
import inspect
import os

from collections import defaultdict
from itertools import chain

class Config:
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

   def __init__(self, **kwargs):
      for k, v in kwargs.items():
         setattr(self, k, v)

   def SPAWN(self):
      R, C = Config.R, Config.C
      spawn, border, sz = [], Config.BORDER, Config.SZ
      spawn += [(border, border+i) for i in range(sz)]
      spawn += [(border+i, border) for i in range(sz)]
      spawn += [(R-1, border+i) for i in range(sz)]
      spawn += [(border+i, C-1) for i in range(sz)]
      idx = np.random.randint(0, len(spawn))
      return spawn[idx]
