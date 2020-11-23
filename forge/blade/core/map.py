from pdb import set_trace as T
import numpy as np

from forge.blade import core
from forge.blade.lib import enums, utils

import os
import time

def loadTiled(tiles, fPath, materials):
    idxMap = np.load(fPath)
    for r, row in enumerate(idxMap):
       for c, idx in enumerate(row):
          mat  = materials[idx]
          tile = tiles[r, c]

          tile.mat      = mat()
          tile.ents     = {}

          tile.state    = mat()
          tile.capacity = tile.mat.capacity
          tile.tex      = mat.tex

          tile.nEnts.update(0)
          tile.index.update(tile.state.index)

class Map:
   def __init__(self, realm, config):
      sz              = config.TERRAIN_SIZE
      self.shape      = (sz, sz)
      self.config     = config

      self.tiles = np.zeros(self.shape, dtype=object)
      for r in range(sz):
         for c in range(sz):
            self.tiles[r, c] = core.Tile(realm, config, enums.Grass, r, c, 'grass')

   def reset(self, realm, idx):
      materials = dict((mat.value.index, mat.value) for mat in enums.Material)
      fName     = self.config.ROOT + str(idx) + self.config.SUFFIX

      loadTiled(self.tiles, fName, materials)
      self.updateList = set()
 
   def harvest(self, r, c):
      self.updateList.add(self.tiles[r, c])
      return self.tiles[r, c].harvest()

   def inds(self):
      return np.array([[j.state.index for j in i] for i in self.tiles])

   def packet(self):
       missingResources = []
       for e in self.updateList:
           missingResources.append(e.pos)
       return missingResources
   
   def step(self):
      for e in self.updateList.copy():
         if e.static:
            self.updateList.remove(e)
         #Perform after check: allow texture to reset
         e.step()

   def stim(self, pos, rng):
      r, c = pos
      rt, rb = r-rng, r+rng+1
      cl, cr = c-rng, c+rng+1
      return self.tiles[rt:rb, cl:cr]

   #Fix this function to key by attr for mat.index 
   def getPadded(self, mat, pos, sz, key=lambda e: e):
      ret = np.zeros((2*sz+1, 2*sz+1), dtype=np.int32)
      R, C = pos
      rt, rb = R-sz, R+sz+1
      cl, cr = C-sz, C+sz+1
      for r in range(rt, rb):
         for c in range(cl, cr):
            if utils.inBounds(r, c, self.size):
               ret[r-rt, c-cl] = key(mat[r, c])
            else:
               ret[r-rt, c-cl] = 0
      return ret

   #This constant re-encode is slow
   def np(self):
      env = np.array([e.state.index for e in 
            self.tiles.ravel()]).reshape(*self.shape)
      return env
