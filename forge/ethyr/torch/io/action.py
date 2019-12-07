'''Action decision module'''

from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

from forge.ethyr.torch.policy import attention
from forge.ethyr.torch.policy import functional

class NetTree(nn.Module):
   def __init__(self, config):
      '''Network responsible for selecting actions

      Args:
         config: A Config object
      '''
      super().__init__()
      self.config = config
      self.h = config.HIDDEN

      self.net = DiscreteAction(self.config, self.h, self.h)

   def names(self, nameMap, args):
      '''Lookup argument indices from name mapping'''
      return np.array([nameMap.get(e) for e in args])

   def forward(self, obs, values, observationTensor, entityLookup, manager):
      '''Populates an IO object with actions in-place                         
                                                                              
      Args:                                                                   
         obs               : An IO object specifying observations
         vals              : A value prediction for each agent
         observationTensor : A fixed size observation representation
         entityLookup      : A fixed size representation of each entity
         manager           : A RolloutManager object
      ''' 
      observationTensor = observationTensor.unsqueeze(-2)
      
      for atn, action in obs.atn.actions.items():
         for arg, data in action.arguments.items():
            #Perform forward pass
            tensor, lens  = data
            targs         = torch.stack([entityLookup[e] for e in tensor])
            atns, atnsIdx = self.net(observationTensor, targs, lens)

            #Gen Atn_Arg style names for backward pass
            name = '_'.join([atn.__name__, arg.__name__])
            if not self.config.TEST:
               manager.collectOutputs(name, obs.keys, atns, atnsIdx, values)

            #Convert from local index over atns to
            #absolute index into entity lookup table
            idxs = atnsIdx.numpy().tolist()
            idxs = [t[a] for t, a in zip(tensor, idxs)]
            obs.atn.actions[atn].arguments[arg] = idxs

class Action(nn.Module):
   '''Head for selecting an action'''
   def forward(self, x, mask=None):
      xIdx = functional.classify(x, mask)
      return x, xIdx

class DiscreteAction(Action):
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

