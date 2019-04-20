from pdb import set_trace as T
import numpy as np


class Tile:
   def __init__(self, config, mat, r, c, nCounts, tex):
      self.r, self.c = r, c
      self.mat = mat()
      self.ents = {}
      self.state = mat()
      self.capacity = self.mat.capacity
      self.counts = np.zeros(nCounts)
      self.tex = tex

      self.inputs(config)

   def inputs(self, config):
      for name, cls in config.static.Tile:
         name = name[0].lower() + name[1:]
         setattr(self, name, cls())

   #@property
   #def nEnts(self):
   #   return len(self.ents)

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

   @property
   def static(self):
      assert self.capacity <= self.mat.capacity
      return self.capacity == self.mat.capacity

   def harvest(self):
      if self.capacity == 0:
         return False
      elif self.capacity <= 1:
         self.state = self.mat.degen()
      self.capacity -= 1
      return True
      return self.mat.dropTable.roll()
