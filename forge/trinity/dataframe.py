from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.io.stimulus import Static

'''
Notes:
1. You need 3d tables for fast row/col range indexing
2. You need a matrix + 2d table representation for entities
3. You need to modify the update() functions to compute absolute
   offsets for discrete variables
4. You need to modify env.getStims and stimulus.Dynamic to accept the new data forms
5. You need to modify RLlib to accept the new data forms
6. You need to modify forge/ethyr/io to accept the new data forms
7. You need to modify the game rules to ensure 1 agent max per tile
'''

class DataType:
   CONTINUOUS = np.float32
   DISCRETE   = np.int32

class Index:
   def __init__(self, prealloc):
      self.free  = {idx for idx in range(1, prealloc)}
      self.index = {}

   def full(self):
      return len(self.free) == 0

   def remove(self, key):
      row = self.index[key]
      del self.index[key]

      self.free.add(row)
      return row

   def update(self, key):
      if key in self.index:
         row = self.index[key]
      else:
         row = self.free.pop()
         self.index[key] = row

      return row

   def get(self, key):
      return self.index[key]

   def expand(self, cur, nxt):
      self.free.update({idx for idx in range(cur, nxt)})

class ContinuousTable:
   def __init__(self, config, obj, prealloc, dtype=DataType.CONTINUOUS):
      self.config = config
      self.dtype  = dtype
      self.cols   = {}
      self.nCols  = 0

      for (attribute,), attr in obj:
         self.initAttr(attribute, attr)

      self.data = self.initData(prealloc, self.nCols)

   def initAttr(self, key, attr):
      if attr.CONTINUOUS:
         self.cols[key] = self.nCols
         self.nCols += 1

   def initData(self, nRows, nCols):
      return np.zeros((nRows, nCols), dtype=self.dtype)

   def update(self, row, attr, val):
      col = self.cols[attr] 
      self.data[row, col] = val

   def expand(self, cur, nxt):
      data       = self.initData(nxt, self.nCols)
      data[:cur] = self.data

      self.data  = data
      self.nRows = nxt

   def get(self, rows, pad=None):
      data = self.data[rows]
      data[rows==0] = 0

      #if pad is not None:
      #   data = np.pad(data, ((0, pad-len(data)), (0, 0)))

      return data

class DiscreteTable(ContinuousTable):
   def __init__(self, config, obj, prealloc, dtype=DataType.DISCRETE):
      self.discrete, self.cumsum = {}, 0
      super().__init__(config, obj, prealloc, dtype)

   def initAttr(self, key, attr):
      if not attr.DISCRETE:
         return

      self.cols[key]     =  self.nCols

      #Flat index
      attr               =  attr(None, None, 0, config=self.config)
      self.discrete[key] =  self.cumsum

      self.cumsum        += attr.max - attr.min + 1
      self.nCols         += 1

   def update(self, row, attr, val):
      col = self.cols[attr] 
      self.data[row, col] = val + self.discrete[attr]

class Grid:
   def __init__(self, R, C):
      self.data = np.zeros((R, C), dtype=np.int)

   def zero(self, pos):
      r, c            = pos
      self.data[r, c] = 0
    
   def set(self, pos, val):
      r, c            = pos
      self.data[r, c] = val
 
   def move(self, pos, nxt, row):
      self.zero(pos)
      self.set(nxt, row)

   def window(self, rStart, rEnd, cStart, cEnd):
      crop = self.data[rStart:rEnd, cStart:cEnd].ravel()
      return crop
      return list(filter(lambda x: x != 0, crop))
      
class GridTables:
   def __init__(self, config, obj, pad, prealloc=1000, expansion=2):
      self.grid       = Grid(config.TERRAIN_SIZE, config.TERRAIN_SIZE)
      self.continuous = ContinuousTable(config, obj, prealloc)
      self.discrete   = DiscreteTable(config, obj, prealloc)
      self.index      = Index(prealloc)

      self.nRows      = prealloc
      self.expansion  = expansion
      self.radius     = config.STIM
      self.pad        = pad

   def get(self, ent, radius=None):
      if radius is None:
         radius = self.radius

      r, c = ent.pos
      cent = self.grid.data[r, c]
      assert cent != 0

      rows = self.grid.window(
            r-radius, r+radius+1,
            c-radius, c+radius+1)

      #Center element first
      #rows.remove(cent)
      #rows.insert(0, cent)
      #This will screw up conv models

      return {'Continuous': self.continuous.get(rows, self.pad),
              'Discrete':   self.discrete.get(rows, self.pad)}

   def update(self, obj, val):
      key, attr = obj.key, obj.attr
      if self.index.full():
         cur        = self.nRows
         self.nRows = cur * self.expansion

         self.index.expand(cur, self.nRows)
         self.continuous.expand(cur, self.nRows)
         self.discrete.expand(cur, self.nRows)

      row = self.index.update(key)
      if obj.DISCRETE:
         self.discrete.update(row, attr, val - obj.min)
      if obj.CONTINUOUS:
         self.continuous.update(row, attr, val)

   def move(self, key, pos, nxt):
      row = self.index.get(key)
      self.grid.move(pos, nxt, row)

   def init(self, key, pos):
      row = self.index.get(key)
      self.grid.set(pos, row)

   def remove(self, key, pos):
      self.index.remove(key)
      self.grid.zero(pos)

class Dataframe:
   def __init__(self, config):
      self.config, self.data = config, defaultdict(dict)
      for (objKey,), obj in Static:
         self.data[objKey] = GridTables(config, obj, pad=obj.N(config))

   def update(self, node, val):
      self.data[node.obj].update(node, val)

   def remove(self, obj, key, pos):
      self.data[obj.__name__].remove(key, pos)

   def init(self, obj, key, pos):
      self.data[obj.__name__].init(key, pos)

   def move(self, obj, key, pos, nxt):
      dat = self.data['Entity'].grid.data.ravel().tolist()
      dat = [e for e in dat if e != 0]
      if len(np.unique(dat)) != len(dat):
         T() 
      self.data[obj.__name__].move(key, pos, nxt)

   def get(self, ent):
      stim = {}
     
      #r, c = ent.pos
      #dat  = self.data['Entity'].index.get(ent.entID)
      #cent = self.data['Entity'].grid.data[r, c]
      #assert dat == cent
      #entDat = self.data['Entity']
      #continuous = entDat.continuous.get(cent)
      #discrete   = entDat.discrete.get(cent)

      stim['Entity'] = self.data['Entity'].get(ent, radius=0)
      stim['Tile']   = self.data['Tile'].get(ent)
      #for key, grid2Table in self.data.items():
      #   stim[key] = grid2Table.get(ent)
      continuous = stim['Entity']['Continuous'][0]
      assert ent.base.self.val         == continuous[0]
      assert ent.base.population.val   == continuous[1]
      assert ent.base.r.val            == continuous[2]
      assert ent.base.c.val            == continuous[3]
      assert ent.history.damage.val    == continuous[4]
      assert ent.history.timeAlive.val == continuous[5]
      assert ent.resources.food.val    == continuous[6]
      assert ent.resources.water.val   == continuous[7]
      assert ent.resources.health.val  == continuous[8]
      assert ent.status.freeze.val     == continuous[9]
      assert ent.status.immune.val     == continuous[10]
      assert ent.status.wilderness.val == continuous[11]
 
      return stim



