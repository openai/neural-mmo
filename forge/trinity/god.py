from pdb import set_trace as T
from forge.blade.lib.enums import Palette
import numpy as np

class God:
   def __init__(self, config, args):
      self.config, self.args = config, args
      self.nEnt, self.nANN = config.NENT, config.NPOP
      self.popSz = self.nEnt // self.nANN
      self.popCounts = np.zeros(self.nANN)
      self.palette = Palette(config.NPOP)
      self.entID = 0

   #Returns IDs for spawning
   def spawn(self):
      entID = str(self.entID)
      annID = hash(entID) % self.nANN
      self.entID += 1

      assert self.popCounts[annID] <= self.popSz
      if self.popCounts[annID] == self.popSz:
         return self.spawn()

      self.popCounts[annID] += 1
      color = self.palette.color(annID)

      return entID, (annID, color)

   def cull(self, annID):
      self.popCounts[annID] -= 1
      assert self.popCounts[annID] >= 0

   def send(self):
      return

   def recv(self, pantheonUpdates):
      return

