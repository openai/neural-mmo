'''Policy submodules and a baseline agent.'''
from pdb import set_trace as T

import numpy as np
from collections import defaultdict

import torch
from torch import nn

from forge import trinity

from forge.ethyr.torch import param
from forge.ethyr.torch.param import zeroGrads

from forge.ethyr.torch import policy
from forge.ethyr.torch.policy import embed, attention
from forge.ethyr.torch.io.action import NetTree
from forge.ethyr.torch.io.stimulus import Env

class Attributes(attention.Attention):
    def __init__(self, config):
      '''Attentional network over attributes

      Args:                                                                   
         config: A Configuration object
      '''
      super().__init__(config.EMBED, config.HIDDEN)
 
class Entities(nn.Module):
    def __init__(self, config):
      '''Attentional network over entities

      Args:                                                                   
         config: A Configuration object
      '''
      super().__init__()
      h = config.HIDDEN 
      self.targDim = 250*h

      self.fc = nn.Linear(self.targDim, 4*h)

    def forward(self, x):
      x = x.view(-1)
      pad = torch.zeros(self.targDim - len(x))
      x = torch.cat([x, pad])
      x = self.fc(x)
      return x

class IO(nn.Module):
    def __init__(self, config):
      #Assemble input network with the IO library
      #Output and value networks
      super().__init__()
      self.input  = Env(config, embed.TaggedInput, Attributes, Entities)
      self.output = NetTree(config)
 
class Hidden(nn.Module):
   def __init__(self, config):
      '''Value network

      Args:                                                                   
         config: A Configuration object
      '''
      super().__init__()
      h = config.HIDDEN
      self.config = config
   
      self.policy = torch.nn.Linear(4*h, h)
      self.value  = torch.nn.Linear(h, 1)
      
   def forward(self, x):
      x = self.policy(x)
 
      val = self.value(x)
      if self.config.TEST:
         val = val.detach()

      return x, val

class Policy(nn.Module):
   def __init__(self, config):
      '''Agent policy

      Args:                                                                   
         config: A Configuration object
      '''
      super().__init__()
      self.config = config

      self.IO = IO(config)

      self.policy = nn.ModuleList([
            Hidden(config) for _ in range(config.NPOP)])

   def forward(self, packet, manager):
      '''Populates an IO object with actions in-place
                                                                              
      Args:                                                                   
         data    : An IO object specifying observations                      
         manager : A RolloutManager object
      ''' 

      #Some shards may not always have data
      if packet.obs.n == 0: return

      #Run the input network
      observationTensor, entityLookup = self.IO.input(packet)

      #Run the main hidden network with unshared population and value weights
      hidden, values = self.hidden(packet, observationTensor)

      #Run the output network
      self.IO.output(packet, values, hidden, entityLookup, manager)

   def hidden(self, packet, state):
      '''Population-specific hidden network and value function

      Args:                                                                   
         packet: An IO object
         state: The current hidden state
 
      Returns:
         hidden: The new hidden state
         values: The value estimate
      ''' 
      idxs, hidden, values = [], [], []
      groups = self.grouped(
            packet.keys,
            state,
            lambda key: key[0])

      for pop, s in groups.items():
         k, s = s
         h, v = self.policy[pop](s)
         idxs.append(k)
         hidden.append(h)
         values.append(v)

      #Rearrange in input order
      idxs   = np.concatenate(idxs).tolist()
      hidden = torch.cat(hidden)[idxs]
      values = torch.cat(values)[idxs]
   
      return hidden, values

   def grouped(self, keys, vals, groupFn):
      '''Group input data by population

      Args:                                                                   
         keys    : Keys
         vals    : Vals
         groupFn : Hash fn
 
      Returns:
         groups: population keyed dictionary of input data
      ''' 

      idx = 0
      groups = defaultdict(lambda: [[], []])
      for key, val in zip(keys, vals):
         key = groupFn(key)
         groups[key][0].append(idx)
         groups[key][1].append(val)
         idx += 1

      for key in groups.keys():
         groups[key][1] = torch.stack(groups[key][1])

      return groups

   def grads(self):
      '''Get gradients and zero out buffers

      Returns:
         grads: A vector of gradients
      ''' 
      grads = param.getGrads(self)
      zeroGrads(self)
      return grads

   def params(self):
      '''Get model parameter vector

      Returns:
         params: A vector of parameters 
      ''' 
      return param.getParameters(self)
