from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.io import action, stimulus

from forge.blade.entity.player import Player
from forge.ethyr.torch import utils


def reverse(f):
    return f.__class__(map(reversed, f.items()))

class Stim:
   def __init__(self, config):
      self.data = []

   def serialize(self, stim, iden):
      rets = {}
      for group, data in stim.items():
         names, data = data
         names = [(iden + e.serial) for e in names]
         rets[group] = (names, data)
      self.data.append(rets)

   def pack(self):
      data = self.data
      self.data = []

      return stimulus.Dynamic.batch(data)
 
class Action:
   def __init__(self, config):
      actions  = action.Dynamic.flat()
      self.idxAtn = dict(enumerate(actions))
      self.atnIdx = reverse(self.idxAtn)
      self.data = []

   def serialize(self, outs, iden):
      ret = []
      for _, out in outs.items():
         arguments, idx = out
         idx = int(out[1])
         args = []
         for e in arguments:
            if type(e) == type and issubclass(e, action.Node):
               args.append(self.atnIdx[e])
            else:
               args.append(e.serial[-1]) #Unique within the training example
         ret.append([args, idx])
      self.data.append(ret)

   def pack(self):
      data = self.data
      self.data = []
      return action.Dynamic.batch(data)

class Serial:
   def __init__(self, config):
      self.config = config

      self.stim    = Stim(config)
      self.action  = Action(config)
      self.keys    = []
      self.reward  = []

   def __len__(self):
      return len(self.keys)

   def serialize(self, env, ent, stim, actions, iden):
      self.stim.serialize(stim, iden)
      self.action.serialize(actions, iden)
      self.keys.append(iden + ent.serial)

   def deserialize(self, keys, stim, action, reward):
      #action   = Action.deserialize(action)
      return keys, stim, action, reward

   def rewards(self, rewards):
      self.reward += rewards

   def pack(self):
      stims   = self.stim.pack()
      actions = self.action.pack()
      keys    = utils.pack(self.keys)
      reward  = self.reward      

      return keys, stims, actions, reward

   def finish(self):
      ret = self.pack()
      self.keys, self.stim = [], []
      self.action = []
      return ret




      
