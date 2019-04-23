from pdb import set_trace as T
import numpy as np
import inspect

from collections import defaultdict
from itertools import chain

class Config:
   ROOT = 'resource/maps/procedural/map'
   SUFFIX = '/map.tmx'

   SZ = 62
   BORDER = 9
   R = C = SZ + BORDER

   STIM = 7
   WINDOW = 2*STIM + 1

   NENT = 256
   NTILE = 6 #Add this to tile static

   RESOURCE = 32
   HEALTH = 10
   IMMUNE = 15

   #Attack ranges
   MELEERANGE = 1
   RANGERANGE = 2
   MAGERANGE  = 3

   MELEEDAMAGE = 10
   RANGEDAMAGE = 2
   MAGEDAMAGE  = 1

   def __init__(self, remote=False, **kwargs):
      from forge.blade.io import stimulus

      self.static = stimulus.Static
      self.dynamic = stimulus.Dynamic()
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


