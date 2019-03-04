from pdb import set_trace as T
import numpy as np

class Config:
   def __init__(self, remote=False, **kwargs):
      self.defaults()
      for k, v in kwargs.items():
         setattr(self, k, v)

      if remote:
         self.ROOT = '/root/code/Projekt-Godsword/' + self.ROOT

   def defaults(self):
      self.ROOT = 'resource/maps/procedural/map'
      self.SUFFIX = '/map.tmx'
      self.BORDER = 9
      self.SZ = 62
      self.R = self.SZ + self.BORDER
      self.C = self.SZ + self.BORDER

      self.STIM = 7
      self.NENT = 256

      #Base agent stats
      self.HEALTH = 10
      self.FOOD = 32
      self.WATER = 32

      #Attack ranges
      self.MELEERANGE = 1
      self.RANGERANGE = 4
      self.MAGERANGE  = 4

   def SPAWN(self):
      R, C = self.R, self.C
      spawn, border, sz = [], self.BORDER, self.SZ
      spawn += [(border, border+i) for i in range(sz)]
      spawn += [(border+i, border) for i in range(sz)]
      spawn += [(R-1, border+i) for i in range(sz)]
      spawn += [(border+i, C-1) for i in range(sz)]
      idx = np.random.randint(0, len(spawn))
      return spawn[idx]

   #Damage formulas. Lambdas don't pickle well
   def MELEEDAMAGE(self, ent, targ): return 0
   def RANGEDAMAGE(self, ent, targ): return 0
   def MAGEDAMAGE(self, ent, targ): return 0

