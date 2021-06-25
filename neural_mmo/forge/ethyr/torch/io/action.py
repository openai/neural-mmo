'''Action decision module'''

from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn

from neural_mmo.forge.blade.io.action import static
from neural_mmo.forge.blade.io.stimulus.static import Stimulus

from neural_mmo.forge.ethyr.torch.policy import attention
from neural_mmo.forge.ethyr.torch.policy import functional

class Output(nn.Module):
   def __init__(self, config):
      '''Network responsible for selecting actions

      Args:
         config: A Config object
      '''
      super().__init__()
      self.config = config
      self.h = config.HIDDEN

      self.net = DiscreteAction(self.config, self.h, self.h)
      self.arg = nn.Embedding(static.Action.n, self.h)
      #self.net = FlatAction(self.config, self.h, self.h)

   def names(self, nameMap, args):
      '''Lookup argument indices from name mapping'''
      return np.array([nameMap.get(e) for e in args])

   def forward(self, obs, lookup):
      '''Populates an IO object with actions in-place                         
                                                                              
      Args:                                                                   
         obs               : An IO object specifying observations
         vals              : A value prediction for each agent
         observationTensor : A fixed size observation representation
         entityLookup      : A fixed size representation of each entity
         manager           : A RolloutManager object
      ''' 
      rets = defaultdict(dict)
      for atn in static.Action.edges:
         for arg in atn.edges:
            lens  = None
            if arg.argType == static.Fixed:
               batch = obs.shape[0]
               idxs  = [e.idx for e in arg.edges]
               #cands = lookup[static.Fixed.__name__][idxs]
               cands = self.arg.weight[idxs]
               cands = cands.repeat(batch, 1, 1)
               #Fixed arg
            else:
               #Temp hack, rename
               cands = lookup['Entity']
               lens  = lookup['N']

            logits = self.net(obs, cands, lens)

            #String names for RLlib for now
            #rets[atn.__name__][arg.__name__] = logits
            rets[atn][arg] = logits

      return rets
      
#Root action class
class Action(nn.Module):
   pass

class DiscreteAction(Action):
   '''Head for making a discrete selection from
   a variable number of candidate actions'''
   def __init__(self, config, xdim, h):
      super().__init__()
      self.net = attention.DotReluBlock(h)

   def forward(self, stim, args, lens):
      x = self.net(stim, args)

      if lens is not None:
         mask = torch.arange(x.shape[-1]).to(x.device).expand_as(x)
         x[mask >= lens] = 0

      return x
