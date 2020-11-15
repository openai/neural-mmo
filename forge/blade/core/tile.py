from pdb import set_trace as T
import numpy as np

from forge.blade.lib import enums
from forge.blade.io.stimulus import Static

def camel(string):
   return string[0].lower() + string[1:]

class Tile:
   SERIAL = 1
   def __init__(self, realm, config, mat, r, c, nCounts, tex):
      self.realm = realm
      self.mat   = mat()
      self.ents  = {}

      self.state    = mat()
      self.capacity = self.mat.capacity
      self.tex      = tex

      self.serialized = 'R' + str(r) + '-C' + str(c)

      self.r     = Static.Tile.R(realm.dataframe, self.serial, r)
      self.c     = Static.Tile.C(realm.dataframe, self.serial, c)
      self.nEnts = Static.Tile.NEnts(realm.dataframe, self.serial)
      self.index = Static.Tile.Index(realm.dataframe, self.serial, self.state.index)

      realm.dataframe.init(Static.Tile, self.serial, (r, c))
 
   @property
   def repr(self):
      return ((self.r, self.c))

   def packet(self):
      data = {}

   @property
   def serial(self):
      return self.serialized

   @property
   def pos(self):
      return self.r.val, self.c.val

   @property
   def impassible(self):
      return self.mat.index in enums.IMPASSIBLE

   @property
   def habitable(self):
      return self.mat.index in enums.HABITABLE

   def addEnt(self, entID, ent):
      assert entID not in self.ents
      self.ents[entID] = ent

   def delEnt(self, entID):
      assert entID in self.ents
      del self.ents[entID]

   def step(self):
      if (not self.static and 
            np.random.rand() < self.mat.respawnProb):
         self.capacity += 1

      #Try inserting a pass
      if self.static:
         self.state = self.mat
         self.index.update(self.state.index)

   @property
   def static(self):
      assert self.capacity <= self.mat.capacity
      return self.capacity == self.mat.capacity

   def harvest(self):
      if self.capacity == 0:
         return False
      elif self.capacity <= 1:
         self.state = self.mat.degen()
         self.index.update(self.state.index)
      self.capacity -= 1
      return True
      return self.mat.dropTable.roll()
