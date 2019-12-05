'''Demo agent class

The policies for v1.2 are sanity checyks only.
I rushed a simple baseline to get the new client
out faster. I am working on something better now.
If you want help getting other models working in
the meanwhile, drop in the Discord support channel.'''
from pdb import set_trace as T
import numpy as np
import torch
import time

from torch import nn

from forge import trinity

from forge.blade.lib.enums import Neon
from forge.blade.lib import enums
from forge.ethyr import torch as torchlib

from forge.ethyr.torch import policy
from forge.blade import entity

from forge.ethyr.torch.param import setParameters, getParameters, zeroGrads
from forge.ethyr.torch import param

from forge.ethyr.torch.policy import functional
from forge.ethyr.io.utils import pack, unpack

from forge.ethyr.torch.io.stimulus import Env
from forge.ethyr.torch.io.action import NetTree
from forge.ethyr.torch.policy import attention

class Entities(nn.Module):
    def __init__(self, h):
      super().__init__()
      self.fc = nn.Linear(235*h, h)

    def forward(self, x):
      x = x.view(-1)
      return self.fc(x)

class ValNet(nn.Module):
   def __init__(self, config):
      super().__init__()
      h = config.HIDDEN
      self.config = config
      self.fc1 = torch.nn.Linear(h, 1)

   def forward(self, stim):
      val = self.fc1(stim)
      if self.config.TEST:
         val = val.detach()

      return val
 
class Net(nn.Module):
   def __init__(self, config):
      super().__init__()
      h = config.HIDDEN

      self.attributes = nn.ModuleDict({
         'Tile':   attention.Attention(config.EMBED, config.HIDDEN),
         'Entity': attention.Attention(config.EMBED, config.HIDDEN)
      })

      #self.entities = attention.Attention(config.HIDDEN, config.HIDDEN)
      self.entities = Entities(config.HIDDEN)
      self.val      = ValNet(config)

      #self.val = nn.ModuleList([ValNet(config)
      #     for _ in range(config.NPOP)])

class ANN(nn.Module):
   def __init__(self, config):
      '''Demo model'''
      super().__init__()
      self.config = config
      self.net    = Net(config)

      #Shared environment/action maps
      self.env    = Env(config)
      self.action = NetTree(config)

   def grouped(self, keys, vals, groupFn):
      '''Group by population'''
      groups = defaultdict(lambda: [[], []])
      for key, val in zip(keys, vals):
         key = groupFn(key)
         groups[key][0].append(key)
         groups[key][1].append(val)

      for key in groups.keys():
         groups[key] = torch.stack(groups[key])

      return groups

   def forward(self, data, manager):
      if data.obs.n == 0:
         return

      observationTensor, entityLookup = self.env(self.net, data)

      #Per pop internal net and value net
      #You have a population group function, but you'll need to reorder
      #To match action ordering
      #for pop, data in self.grouped(obs.keys, observationTensor):
      #   keys, obsTensor = data
      #   self.net.val[pop](obsTensor)
         
      vals = self.net.val(observationTensor)
      self.action(data, vals, observationTensor, entityLookup, manager)

   def grads(self):
      grads = param.getGrads(self)
      zeroGrads(self)
      return grads

   def params(self):
      return param.getParameters(self)
