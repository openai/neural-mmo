from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from forge.blade.io import stimulus, action
from forge.ethyr.torch.modules import Transformer

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
         x = x.float().view(-1, 1)
      x = self.embed(x)
      return x

class Env(nn.Module):
   def __init__(self, net, config):
      super().__init__()
      h = config.HIDDEN
      self.config = config

      self.net = net
      self.init(config)

   def init(self, config, name=None):
      emb  = nn.ModuleDict()
      for name, subnet in config.static:
         emb[name] = nn.ModuleDict()
         for param, val in subnet:
            emb[name][param] = Input(val(config), config)
      self.emb = emb

   def forward(self, env, ent):
      stims = self.config.dynamic(env, ent, flat=True)
      features, embed = {}, {}
      for group, stim in stims.items():
         names, subnet = stim
         feats = []
         for param, val in subnet.items():
            val = torch.Tensor(val)
            emb = self.emb[group][param](val)
            feats.append(emb.split(1))
         emb = np.array(feats).T.tolist()
         emb = [torch.cat(e) for e in emb]

         feats = torch.stack(emb)
         emb = self.net(torch.stack(emb)).split(1)
         embed = {**embed, **dict(zip(names, emb))}

      features = torch.stack(list(embed.values()), 1)
      features = self.net(features)
      return features, embed
