'''Infrastructure layer for representing agent observations

Maintains a synchronized + serialized representation of agent observations in
flat tensors. This allows for fast observation processing as a set of tensor
slices instead of a lengthy traversal over hundreds of game properties.

Synchronization bugs are notoriously difficult to track down: make sure
to follow the correct instantiation protocol, e.g. as used for defining
agent/tile observations, when adding new types observations to the code'''

from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import nmmo

class DataType:
   CONTINUOUS = np.float32
   DISCRETE   = np.int32

class Index:
   '''Lookup index of attribute names'''
   def __init__(self, prealloc):
      self.free  = {idx for idx in range(1, prealloc)}
      self.index = {}
      self.back  = {}

   def full(self):
      return len(self.free) == 0

   def remove(self, key):
      row = self.index[key]
      del self.index[key]
      del self.back[row]

      self.free.add(row)
      return row

   def update(self, key):
      if key in self.index:
         row = self.index[key]
      else:
         row = self.free.pop()
         self.index[key] = row
         self.back[row]  = key

      return row

   def get(self, key):
      return self.index[key]

   def teg(self, row):
      return self.back[row]

   def expand(self, cur, nxt):
      self.free.update({idx for idx in range(cur, nxt)})

class ContinuousTable:
   '''Flat tensor representation for a set of continuous attributes'''
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

      if pad is not None:
         data = np.pad(data, ((0, pad-len(data)), (0, 0)))

      return data

class DiscreteTable(ContinuousTable):
   '''Flat tensor representation for a set of discrete attributes'''
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
   '''Flat representation of tile/agent positions'''
   def __init__(self, R, C):
      self.data = np.zeros((R, C), dtype=np.int32)

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
      return list(filter(lambda x: x != 0, crop))
      
class GridTables:
   '''Combines a Grid + Index + Continuous and Discrete tables

   Together, these data structures provide a robust and efficient
   flat tensor representation of an entire class of observations,
   such as agents or tiles'''
   def __init__(self, config, obj, pad, prealloc=1000, expansion=2):
      self.grid       = Grid(config.MAP_SIZE, config.MAP_SIZE)
      self.continuous = ContinuousTable(config, obj, prealloc)
      self.discrete   = DiscreteTable(config, obj, prealloc)
      self.index      = Index(prealloc)

      self.nRows      = prealloc
      self.expansion  = expansion
      self.radius     = config.PLAYER_VISION_RADIUS
      self.pad        = pad

   def get(self, ent, radius=None, entity=False):
      if radius is None:
         radius = self.radius

      r, c = ent.pos
      cent = self.grid.data[r, c]

      if __debug__:
          assert cent != 0

      rows = self.grid.window(
            r-radius, r+radius+1,
            c-radius, c+radius+1)

      #Self entity first
      if entity:
         rows.remove(cent)
         rows.insert(0, cent)

      values = {'Continuous': self.continuous.get(rows, self.pad),
                'Discrete':   self.discrete.get(rows, self.pad)}

      if entity:
         ents = [self.index.teg(e) for e in rows]
         if __debug__:
             assert ents[0] == ent.entID
         return values, ents

      return values

   def getFlat(self, keys):
      if __debug__:
          err = f'Dataframe got {len(keys)} keys with pad {self.pad}'
          assert len(keys) <= self.pad, err

      rows = [self.index.get(key) for key in keys[:self.pad]]
      values = {'Continuous': self.continuous.get(rows, self.pad),
                'Discrete':   self.discrete.get(rows, self.pad)}
      return values

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
      if pos is None:
          return

      row = self.index.get(key)
      self.grid.set(pos, row)

   def remove(self, key, pos):
      self.index.remove(key)
      self.grid.zero(pos)

class Dataframe:
   '''Infrastructure wrapper class'''
   def __init__(self, realm):
      config      = realm.config
      self.config = config
      self.data   = defaultdict(dict)

      for (objKey,), obj in nmmo.Serialized:
         if not obj.enabled(config):
             continue
         self.data[objKey] = GridTables(config, obj, pad=obj.N(config))

      self.realm = realm

   def update(self, node, val):
      self.data[node.obj].update(node, val)

   def remove(self, obj, key, pos):
      self.data[obj.__name__].remove(key, pos)

   def init(self, obj, key, pos):
      self.data[obj.__name__].init(key, pos)

   def move(self, obj, key, pos, nxt):
      self.data[obj.__name__].move(key, pos, nxt)

   def get(self, ent):
      stim = {}
     
      stim['Entity'], ents = self.data['Entity'].get(ent, entity=True)
      stim['Entity']['N']  = np.array([len(ents)], dtype=np.int32)

      ent.targets          = ents
      stim['Tile']         = self.data['Tile'].get(ent)
      stim['Tile']['N']    = np.array([self.config.PLAYER_VISION_DIAMETER], dtype=np.int32)

      #Current must have the same pad
      if self.config.ITEM_SYSTEM_ENABLED:
         items                = ent.inventory.dataframeKeys
         stim['Item']         = self.data['Item'].getFlat(items)
         stim['Item']['N']    = np.array([len(items)], dtype=np.int32)

      if self.config.EXCHANGE_SYSTEM_ENABLED:
         market               = self.realm.exchange.dataframeKeys
         stim['Market']       = self.data['Item'].getFlat(market)
         stim['Market']['N']  = np.array([len(market)], dtype=np.int32)

      return stim
