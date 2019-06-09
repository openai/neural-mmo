from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from itertools import chain

from forge.blade.io import stimulus, action
from forge.blade.io import utils
from forge.blade.io.serial import Serial

def reverse(f):
    return f.__class__(map(reversed, f.items()))

class Lookup:
   def __init__(self):
      self.names = []
      self.data  = []

   def add(self, names, data):
      self.names += names
      self.data  += data

   def table(self):
      names, data = self.names, self.data
      dat = dict(zip(names, data))
      names = set(self.names)

      idxs, data = {}, []
      for idx, name in enumerate(names):
         idxs[name] = idx
         data.append(dat[name])

      data = torch.cat(data)
      assert len(idxs) == len(data)
      return idxs, data
 
class Embedding(nn.Module):
   def __init__(self, var, dim):
      super().__init__()
      self.embed = torch.nn.Embedding(var.range, dim)
      self.min = var.min

   def forward(self, x):
      return self.embed(x - self.min)

class Input(nn.Module):
   def __init__(self, cls, config):
      super().__init__()
      self.cls = cls
      if isinstance(cls, stimulus.node.Discrete):
         self.embed = Embedding(cls, config.EMBED)
      elif isinstance(cls, stimulus.node.Continuous):
         self.embed = torch.nn.Linear(1, config.EMBED)

   def forward(self, x):
      if isinstance(self.cls, stimulus.node.Discrete):
         x = x.long()
      elif isinstance(self.cls, stimulus.node.Continuous):
         x = x.float().unsqueeze(2)
      x = self.embed(x)
      return x

class Env(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.config = config
      h = config.HIDDEN
      self.h = h

      self.initSubnets(config)
      self.initActions()

   def initSubnets(self, config, name=None):
      emb  = nn.ModuleDict()
      for name, subnet in config.static:
         emb[name] = nn.ModuleDict()
         for param, val in subnet:
            emb[name][param] = Input(val(config), config)
      self.emb = emb

   def initActions(self):
      self.action = nn.Embedding(action.Static.n, self.h)

   #Embed actions
   def actions(self, lookup):
      for atn in action.Static.actions:
         #Brackets on atn.serial?
         idx = torch.Tensor([atn.idx])
         idx = idx.long().to(self.config.DEVICE)

         emb = self.action(idx)
         #Dirty hack -- remove duplicates
         lookup.add([atn], [emb])

         key = Serial.key(atn, tuple([]))
         #What to replace serial with here
         lookup.add([key], [emb])

   #Pack attributes of each entity
   def attrs(self, group, net, subnet):
      feats = []
      for param, val in subnet.items():
         if param not in 'Thunk Index'.split():
            continue
         val = torch.Tensor(val).to(self.config.DEVICE)
         emb = self.emb[group][param](val)
         feats.append(emb)

      emb = torch.stack(feats, -2)
      emb = net.fc1(emb).squeeze(-2)
      #emb = net(emb)
      return emb

   #Todo: check gpu usage
   def forward(self, net, stims):
      features, lookup = {}, Lookup()
      self.actions(lookup)

      #Pack entities of each observation set
      for group, stim in stims.items():
         names, subnet = stim
         emb = self.attrs(group, net, subnet)
         features[group] = emb.unsqueeze(0)

         #Unpack and flatten for embedding
         lens = [len(e) for e in names]
         vals = utils.unpack(emb, lens, dim=1)
         for k, v in zip(names, vals):
            v = v.split(1, dim=0)
            lookup.add(k, v)

      #Concat feature block
      features = list(features.values())
      features = torch.cat(features, -2)

      #For linear
      features = torch.split(features, 1, dim=-2)
      features = torch.cat(features, -1)
      features = net.fc2(features).squeeze(-2).squeeze(0)

      #features = net(features).squeeze(0)

      embed = lookup.table()
      return features, embed


