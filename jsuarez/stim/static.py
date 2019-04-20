from pdb import set_trace as T
import numpy as np

from forge.ethyr.stim import node

class Static:
   def __init__(self, config):
      self.config = config

      self.data = {}
      self.data['tile']   = self.tile()
      self.data['entity'] = self.entity()

   @property
   def flat(self):
      return self.data

   def entity(self):
      config, stim = self.config, {}

      stim['health']   = node.Continuous(config.HEALTH)
      stim['food']     = node.Continuous(config.FOOD)
      stim['water']    = node.Continuous(config.WATER)

      stim['lifetime'] = node.Continuous()
      stim['damage']   = node.Continuous()
      stim['freeze']   = node.Continuous()

      stim['sameColor'] = node.Discrete(1)

      stim['r'] = node.Discrete(config.R)
      stim['c'] = node.Discrete(config.C)

      STIM = config.STIM
      stim['rDelta'] = node.Discrete(min=-STIM, max=STIM)
      stim['cDelta'] = node.Discrete(min=-STIM, max=STIM)

      return stim

   def tile(self):
      config, stim = self.config, {}

      stim['index'] = node.Discrete(config.NPOP)
      stim['nEnts'] = node.Discrete(config.NENT)

      stim['r'] = node.Discrete(config.R + config.BORDER)
      stim['c'] = node.Discrete(config.C + config.BORDER)

      return stim

