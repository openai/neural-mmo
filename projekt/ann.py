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

#Hacky inner attention layer
class EmbAttn(nn.Module):
   def __init__(self, config, n):
      super().__init__()
      self.fc1 = nn.Linear(n*config.EMBED, config.HIDDEN)

   def forward(self, x):
      batch, ents, _, _ = x.shape
      x = x.view(batch, ents, -1)
      x = self.fc1(x)
      return x

#Hacky outer attention layer
class EntAttn(nn.Module):
   def __init__(self, xDim, yDim, n):
      super().__init__()
      self.fc1 = nn.Linear(n*xDim, yDim)

   def forward(self, x):
      #batch, ents, _, = x.shape
      #x = x.view(batch, -1)
      x = self.fc1(x)
      return x

#Hacky outer attention layer
class TileConv(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.h = h

      self.conv1 = nn.Conv2d(h, h, 3)
      self.pool1 = nn.MaxPool2d(2)

      #self.conv2 = nn.Conv2d(h, h, 3)
      #self.pool2 = nn.MaxPool2d(2)
 
      self.fc1 = nn.Linear(h*6*6, h)
      #self.fc1 = nn.Linear(h*2*2, h)

   def forward(self, x):
      x = x.transpose(-2, -1)
      x = x.view(-1, self.h, 15, 15)
      x = self.pool1(self.conv1(x))
      #x = self.pool1(torch.relu(self.conv1(x)))
      #x = self.pool2(torch.relu(self.conv2(x)))

      batch, _, _, _ = x.shape
      x = x.view(batch, -1)
      x = self.fc1(x).squeeze(0)

      return x

#Variable number of entities
class Entity(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.emb = attention.Attention(config.EMBED, config.HIDDEN)
      self.ent = attention.Attention(config.HIDDEN, config.HIDDEN)

      #self.emb = attention.Attention2(config.EMBED, config.HIDDEN)
      #self.ent = attention.Attention2(config.HIDDEN, config.HIDDEN)

#Fixed number of entities
class Tile(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.emb = attention.Attention(config.EMBED, config.HIDDEN)
      #self.ent = attention.Attention(config.EMBED, config.HIDDEN)
      #self.emb = attention.Attention2(config.EMBED, config.HIDDEN)
      #self.ent = TileConv(config.HIDDEN)
      self.ent  = attention.FactorizedAttention(config.EMBED, config.HIDDEN, 16)

class Meta(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.fc = nn.Linear(2*h, h)

   def forward(self, x):
      return self.fc(x)
 
#Fixed number of entities
class Attn(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.emb = attention.Attention(config.EMBED, config.HIDDEN)
      #self.emb = attention.FactorizedAttention(config.EMBED, config.HIDDEN, 2)
      #self.ent  = attention.FactorizedAttention(config.EMBED, config.HIDDEN, 16)
      self.ent = attention.Attention(config.EMBED, config.HIDDEN)
      self.tile = TileConv(config.HIDDEN)
      self.meta = Meta(config.HIDDEN)

class Net(nn.Module):
   def __init__(self, config):
      super().__init__()
      h = config.HIDDEN

      #self.attn = Attn(config)
      #self.val  = torch.nn.Linear(h, 1)

      self.attributes = nn.ModuleDict({
         'Tile':   attention.Attention(config.EMBED, config.HIDDEN),
         'Entity': attention.Attention(config.EMBED, config.HIDDEN)
      })

      self.entities = attention.Attention(config.HIDDEN, config.HIDDEN)

      self.val = torch.nn.Linear(h, 1)

      #self.val = nn.ModuleList([ValNet(config)
      #     for _ in range(config.NPOP)])

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

      return stim, val
 
class ANN(nn.Module):
   def __init__(self, config):
      '''Demo model'''
      super().__init__()
      self.config = config
      #self.net = nn.ModuleList([Net(config)
      #      for _ in range(config.NPOP)])

      self.net = Net(config)

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

   def forward(self, obs, manager):
      observationTensor, entityLookup = self.env(self.net, obs)

      #Per pop internal net and value net
      #You have a population group function, but you'll need to reorder
      #To match action ordering
      #for pop, data in self.grouped(obs.keys, observationTensor):
      #   keys, obsTensor = data
      #   self.net.val[pop](obsTensor)
         
      vals = self.net.val(observationTensor)
      self.action(obs, vals, observationTensor, entityLookup, manager)
      return vals

   def recvUpdate(self, update):
      setParameters(self, update)

   def grads(self):
      grads = param.getGrads(self)
      zeroGrads(self)
      return grads

   def params(self):
      return param.getParameters(self)
