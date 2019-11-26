from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn

from forge.ethyr.torch.policy import attention
from forge.ethyr.torch.policy import functional
from forge.blade.io import Action as Static
from forge.blade.io.action import static as action

from forge.ethyr.io import Action as Dynamic
from forge.ethyr.io.utils import pack, unpack

from forge.blade.entity.player import Player

class NetTree(nn.Module):
   '''Network responsible for selecting actions

   Args:
      config: A Config object
   '''
   def __init__(self, config):
      super().__init__()
      self.config = config
      self.h = config.HIDDEN

      self.net = VariableDiscreteAction(
               self.config, self.h, self.h)

   def names(self, nameMap, args):
      return np.array([nameMap.get(e) for e in args])

   def forward(self, obs, values, observationTensor, entityLookup, manager):
      observationTensor = observationTensor.unsqueeze(-2)
      
      for atn, action in obs.atn.actions.items():
         for arg, data in action.arguments.items():
            #Perform forward pass
            tensor, lens  = data
            vals          = torch.stack([entityLookup[e] for e in tensor])
            atns, atnsIdx = self.net(observationTensor, vals, lens)

            #Gen Atn_Arg style names for backward pass
            name = '_'.join([atn.__name__, arg.__name__])
            manager.collectOutputs(name, obs.keys, atns, atnsIdx, values)

            #Convert from local index over atns to
            #absolute index into entity lookup table
            atnsIdx = atnsIdx.numpy().tolist()
            atnsIdx = [t[a] for t, a in zip(tensor, atnsIdx)]
            obs.atn.actions[atn].arguments[arg] = [atns, atnsIdx]

class Action(nn.Module):
   '''Head for selecting an action'''
   def forward(self, x, mask=None):
      xIdx = functional.classify(x, mask)
      return x, xIdx

class ConstDiscreteAction(Action):
   '''Head for making a discrete selection from
   a constant number of candidate actions'''
   def __init__(self, config, h, ydim):
      super().__init__()
      self.net = torch.nn.Linear(h, ydim)

   def forward(self, stim):
      x = self.net(stim)
      if len(x.shape) > 1:
         x = x.squeeze(-2)
      return super().forward(x)

class VariableDiscreteAction(Action):
   '''Head for making a discrete selection from
   a variable number of candidate actions'''
   def __init__(self, config, xdim, h):
      super().__init__()
      self.net = attention.DotReluBlock(h)

   def forward(self, stim, args, lens):
      x = self.net(stim, args)

      lens      = torch.LongTensor(lens).unsqueeze(-1)
      n, maxLen = x.shape[0], x.shape[-1]

      inds = torch.arange(maxLen).expand_as(x)
      mask = inds < lens 

      return super().forward(x, mask)

