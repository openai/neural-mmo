'''Policy submodules and a baseline agent.'''
from pdb import set_trace as T

import torch
from torch import nn

from forge import trinity

from forge.ethyr.torch import param
from forge.ethyr.torch.param import zeroGrads

from forge.ethyr.torch import policy
from forge.ethyr.torch.policy import embed, attention
from forge.ethyr.torch.io.action import NetTree
from forge.ethyr.torch.io.stimulus import Env

class ValNet(nn.Module):
   def __init__(self, config):
      '''Value network

      Args:                                                                   
         config: A Configuration object
      '''
      super().__init__()
      h = config.HIDDEN
      self.config = config
      self.fc1 = torch.nn.Linear(h, 1)

   def forward(self, stim):
      val = self.fc1(stim)
      if self.config.TEST:
         val = val.detach()

      return val

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
      self.fc1 = nn.Linear(4*h, h)

    def forward(self, x):
      x = x.view(-1)
      pad = torch.zeros(self.targDim - len(x))
      x = torch.cat([x, pad])
      x = self.fc(x)
      x = torch.relu(x)
      x = self.fc1(x)
      return x

class ANN(nn.Module):
   def __init__(self, config):
      '''Agent policy

      Args:                                                                   
         config: A Configuration object
      '''
      super().__init__()
      self.config = config

      #Module references
      embeddings = embed.TaggedInput
      attributes = Attributes
      entities   = Entities

      #Assemble input network with the IO library
      self.env    = Env(config, embeddings, attributes, entities)

      #Output and value networks
      self.action = NetTree(config)
      self.val    = ValNet(config)

   def forward(self, data, manager):
      '''Populates an IO object with actions in-place
                                                                              
      Args:                                                                   
         data    : An IO object specifying observations                      
         manager : A RolloutManager object
      ''' 

      #Some shards may not always have data
      if data.obs.n == 0: return

      observationTensor, entityLookup = self.env(data)
      vals                            = self.val(observationTensor)

      self.action(data, vals, observationTensor, entityLookup, manager)

   def grouped(self, keys, vals, groupFn):
      '''Group input data by population

      Returns:                                                                            groups: population keyed dictionary of input data
      ''' 
      #Per pop internal net and value net
      #You have a population group function, but you'll need to reorder
      #To match action ordering
      #for pop, data in self.grouped(obs.keys, observationTensor):
      #   keys, obsTensor = data
      #   self.net.val[pop](obsTensor)
 
      groups = defaultdict(lambda: [[], []])
      for key, val in zip(keys, vals):
         key = groupFn(key)
         groups[key][0].append(key)
         groups[key][1].append(val)

      for key in groups.keys():
         groups[key] = torch.stack(groups[key])

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
