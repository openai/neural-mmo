'''Policy submodules and a baseline agent.'''
from pdb import set_trace as T

import numpy as np
from collections import defaultdict

import torch
from torch import nn

from forge import trinity

from forge.ethyr.torch import policy
from forge.ethyr.torch import param
from forge.ethyr.torch import io

class Hidden(nn.Module):
   def __init__(self, config):
      '''Hidden and value networks

      Args:                                                                   
         config: A Configuration object
      '''
      super().__init__()
      h = config.HIDDEN
      self.config = config
   
      #Added policy head
      #self.fc1 = torch.nn.Linear(h, 4*h)
      #self.fc2 = torch.nn.Linear(4*h, h)
      self.value  = torch.nn.Linear(h, 1)
      
   def forward(self, x):
      out = x
      #out = self.fc1(x)
      #out = torch.relu(out)
      #out = self.fc2(out)

      #Note: moved value to end of net
      val = self.value(out)

      if self.config.TEST:
         val = val.detach()

      return out, val

class Policy(nn.Module):
   def __init__(self, config):
      '''Agent policy

      Args:                                                                   
         config: A Configuration object
      '''
      super().__init__()
      self.config = config

      self.IO     = IO(config)

      self.policy = nn.ModuleList([
            Hidden(config) for _ in range(config.NPOP)])

   def forward(self, packet, manager):
      '''Populates an IO object with actions in-place
                                                                              
      Args:                                                                   
         packet  : An IO object specifying observations                      
         manager : A RolloutManager object
      ''' 

      #Some shards may not always have data
      if packet.obs.n == 0: return

      #Run the input network
      observationTensor, entityLookup = self.IO.input(packet)

      #Run the main hidden network with unshared population and value weights
      hidden, values = self.hidden(packet, observationTensor)
      #hidden, values = self.policy[0](observationTensor)

      #Run the output network
      self.IO.output(packet, values, hidden, entityLookup, manager)

   def hidden(self, packet, state):
      '''Population-specific hidden network and value function

      Args:                                                                   
         packet : An IO object specifying observations
         state  : The current hidden state
 
      Returns:
         hidden : The new hidden state
         values : The value estimate
      ''' 

      #Rearrange by population membership
      groups = self.grouped(
            packet.keys,
            state,
            lambda key: key[0])

      #Initialize output buffers
      hidden = torch.zeros((packet.obs.n, self.config.HIDDEN))
      values = torch.zeros((packet.obs.n, 1))

      #Per-population policies rearranged in input order
      for pop in groups:
         idxs, s = groups[pop]
         h, v    = self.policy[pop](s)

         hidden[idxs] = h
         values[idxs] = v 
  
      return hidden, values

   def sanity(self, observationTensor):
      hidden, values = [], []
      for idx, obs in enumerate(observationTensor):
         pop  = packet.keys[idx][0]
         h, v = self.policy[pop](obs)
         hidden.append(h)
         values.append(v)
      hidden = torch.stack(hidden)
      values = torch.stack(values)
      return hidden, values

   def grouped(self, keys, vals, groupFn):
      '''Group input data by population

      Args:                                                                   
         keys    : IO object entity keys
         vals    : Predictions from the value network
         groupFn : Entity key -> population hash function
 
      Returns:
         groups  : Population keyed dictionary of input data
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
      param.zeroGrads(self)
      return grads

   def params(self):
      '''Get model parameter vector

      Returns:
         params: A vector of parameters 
      ''' 
      return param.getParameters(self)
