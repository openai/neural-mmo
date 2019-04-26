from pdb import set_trace as T
import numpy as np

from forge.blade import core
from forge.blade.lib import enums, utils

def loadTiled(config, fPath, tiles, nCounts):
    import pytmx
    tm = pytmx.TiledMap(fPath)
    assert len(tm.layers) == 1
    layer = tm.layers[0]
    W, H = layer.width, layer.height
    tilemap = np.zeros((H, W), dtype=object)
    for w, h, dat in layer.tiles():
       f = dat[0]
       tex = f.split('/')[-1].split('.')[0]
       tilemap[h, w] = core.Tile(config, tiles[tex], h, w, nCounts, tex)
    return tilemap

class Map:
   def __init__(self, config, idx):
      self.updateList = set()
      self.config = config
      self.nCounts = config.NPOP
      self.genEnv(config.ROOT + str(idx) + config.SUFFIX)

   def harvest(self, r, c):
      self.updateList.add(self.tiles[r, c])
      return self.tiles[r, c].harvest()

   def inds(self):
      return np.array([[j.state.index for j in i] for i in self.tiles])

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

   def np(self):
      env = np.array([e.state.index for e in 
            self.tiles.ravel()]).reshape(*self.shape)
      return env
     
   def genEnv(self, fName):
      tiles = dict((mat.value.tex, mat.value) for mat in enums.Material)
      self.tiles = loadTiled(self.config, fName, tiles, self.nCounts)
      self.shape = self.tiles.shape
       
