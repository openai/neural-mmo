from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from itertools import chain

from forge.blade.io import stimulus, action
from forge.ethyr.torch.modules import Transformer
from forge.ethyr.torch import utils

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
   def __init__(self, config, device, mapActions):
      super().__init__()
      h = config.HIDDEN
      self.h = h

      self.config = config
      self.device = device

      self.init(config)
      self.initAction(mapActions)

   def initAction(self, mapActions):
      self.mapActions = mapActions
      actions = action.Dynamic.flat()
      nAtn = len(actions)
      if mapActions:
         self.atnIdx = dict((atn, idx) for idx, atn in enumerate(actions))
      else:
         self.atnIdx = dict((idx, idx) for idx, atn in enumerate(actions))

      assert len(self.atnIdx) == nAtn
      self.action = nn.Embedding(nAtn, self.h)

   def init(self, config, name=None):
      emb  = nn.ModuleDict()
      for name, subnet in config.static:
         emb[name] = nn.ModuleDict()
         for param, val in subnet:
            emb[name][param] = Input(val(config), config)
      self.emb = emb

   #Todo: check gpu usage
   def forward(self, net, stims):
      features, embed = {}, {}
      for atn, idx in self.atnIdx.items():
         idx = torch.Tensor([idx]).long().to(self.device)
         embed[atn] = self.action(idx)

      #Pack entities of each observation set
      for group, stim in stims.items():
         #Names here are flat
         names, subnet = stim
         feats = []

         #Pack attributes of each entity
         for param, val in subnet.items():
            val, lens = utils.pack(val)
            val = val.to(self.device)

            emb = self.emb[group][param](val)
            feats.append(emb)

         emb = torch.stack(feats, 2)
         emb = net(emb)

         features[group] = emb.unsqueeze(0)

         #Flatten for embed
         vals = []
         emb = utils.unpack(emb, lens)
         for e in emb:
            vals += e.split(1, dim=0)
      
         #Store embeddings for action selection later
         embed = {**embed, **dict(zip(names, vals))}

      #Concat feature block
      features = list(features.values())
      features = torch.cat(features, -2)
   
      features = net(features).squeeze(0)
      return features, embed
