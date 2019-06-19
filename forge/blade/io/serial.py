from pdb import set_trace as T
import numpy as np

from forge.blade.io import action, stimulus
from forge.ethyr.torch import utils

#Wrapper class around stimulus/action
#batching and serialization
class Serial:
   KEYLEN = 5
   def __init__(self, config):
      self.config = config
      self.reset()

   def __len__(self):
      return len(self.keys)

   def rewards(self, rewards):
      self.reward += rewards

   def reset(self):
      self.stim, self.action = [], []
      self.keys, self.reward = [], []

   def key(key, iden):
      from forge.blade.entity import Player
      from forge.blade.core.tile import Tile

      ret = key.serial
      if isinstance(key, type):
         if issubclass(key, action.Node):
            ret += tuple([2])
      else:
         ret = iden + key.serial
         if isinstance(key, Player):
            ret += tuple([0])
         elif isinstance(key, Tile):
            ret += tuple([1])

      pad = Serial.KEYLEN - len(ret)
      ret = tuple(pad*[-1]) + ret
      return ret
      #action: 2
      #tile: 1
      #player: 0

   def unkey(self):
      pass
 
   def serialize(self, realm, env, ent, stim, outs):
      iden = realm.worldIdx, realm.tick
      self.stim.append(stimulus.Dynamic.serialize(stim, iden))
      self.action.append(action.Dynamic.serialize(outs, iden))
      self.keys.append(Serial.key(ent, iden))

   def finish(self):
      stims   = stimulus.Dynamic.batch(self.stim)
      atnArgs = action.Dynamic.batch(self.action)
      keys    = utils.pack(self.keys)
      reward  = self.reward      

      self.reset()
      return keys, stims, atnArgs, reward
