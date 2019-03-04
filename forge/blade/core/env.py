#Main world definition. Defines and manages entity handlers,
#Defines behavior of the world under all circumstances and handles
#interaction by agents. Also defines an accurate stimulus that
#encapsulates the world as seen by a particular agent

import numpy as np

from forge.blade import systems
from forge.blade.lib import utils
from forge.blade import lib

from forge.blade import core
from forge.blade.item import rawfish, knife, armor
from pdb import set_trace as T

class Env:
   def __init__(self, config, idx):
      #Load the world file
      self.env = core.Map(config, idx)
      self.shape = self.env.shape
      self.spawn = config.SPAWN
      self.config = config

      #Entity handlers
      self.stimSize = 3
      self.worldDim = 2*self.stimSize+1

      #Exchange - For future updates
      self.market = systems.Exchange()
      sardine = rawfish.Sardine
      nife = knife.Iron
      armr = armor.Iron
      self.market.buy(sardine, 10, 30)
      self.market.sell(nife, 20, 50)
      self.market.buy(armr, 1, 500)
      self.market.sell(armr, 3, 700)

      self.stats = lib.StatTraker()

      self.tick = 0
      self.envTimer  = utils.BenchmarkTimer()
      self.entTimer  = utils.BenchmarkTimer()
      self.cpuTimer  = utils.BenchmarkTimer()
      self.handlerTimer = utils.BenchmarkTimer()
      self.statTimer  = utils.BenchmarkTimer()

   def stim(self, pos):
      return self.env.getPadded(self.env.tiles, pos, 
            self.stimSize, key=lambda e:e.index).astype(np.int8)

   #Hook for render
   def graphicsData(self):
      return self.env, self.stats

   def step(self, pcs, npcs):
      self.stats.update(pcs, npcs, self.market)
      self.tick += 1

