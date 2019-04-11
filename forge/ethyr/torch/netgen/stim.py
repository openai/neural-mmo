from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from forge.blade.action.tree import ActionTree
from forge.blade.action import action
from forge.blade.action.action import ActionRoot, NodeType

from forge.ethyr.stim import node
from forge.ethyr.stim.static import Static

from collections import defaultdict

class Embedding(nn.Module):
   def __init__(self, var, dim):
      super().__init__()
      self.embed = torch.nn.Embedding(var.range, dim)
      self.min = var.min

   def forward(self, x):
      return self.embed(x - self.min)

class Input(nn.Module):
   def __init__(self, val, config):
      super().__init__()
      self.cls = type(val)
      if self.cls is node.Discrete:
         self.embed = Embedding(val, config.EMBED)
      elif self.cls is node.Continuous:
         self.embed = torch.nn.Linear(1, config.EMBED)

   def forward(self, x):
      if self.cls is node.Discrete:
         x = x.long()
      elif self.cls is node.Continuous:
         x = x.float().view(-1, 1)
      x = self.embed(x)
      return x

class Env(nn.Module):
   def __init__(self, config):
      super().__init__()
      h = config.HIDDEN
      self.embeddings = {}

      stim = Static(config).flat
      self.emb, self.proj  = self.init(config, stim)

   def init(self, config, stim, name=None):
      emb  = nn.ModuleDict()
      proj = nn.ModuleDict()
      for name, subnet in stim.items():
         n = 0
         emb[name] = nn.ModuleDict()
         for param, val in subnet.items():
            n += config.EMBED
            emb[name][param] = Input(val, config)
         proj[name] = torch.nn.Linear(n, config.HIDDEN)
      return emb, proj

   def merge(self, dicts):
      ret = defaultdict(list)
      for name, d in dicts.items(): 
         for k, v in d.items():
            ret[k] += [v]
      return ret

   def forward(self, env, ent, stim):
      project, embed = {}, {}
      #add some player keys
      for name, subnet in stim.items():
         merged = self.merge(subnet)
         feats = []
         for param, val in merged.items():
            val = torch.Tensor(val)
            feats.append(self.emb[name][param](val))
         feats = torch.cat(feats, 1)
         proj = self.proj[name](feats)
         project[name] = proj
         zipped = dict(zip(subnet.keys(), proj.split(1)))
         embed = {**embed, **zipped}
      return project, embed
