from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from forge.blade.io.action.dynamic import ActionTree

from forge.blade.io.stim import static, node
from forge.blade.io.action.static import ActionRoot, NodeType

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
      if issubclass(cls, node.Discrete):
         self.embed = Embedding(cls, config.EMBED)
      elif issubclass(cls, node.Continuous):
         self.embed = torch.nn.Linear(1, config.EMBED)

   def forward(self, x):
      if issubclass(self.cls, node.Discrete):
         x = x.long()
      elif issubclass(self.cls, node.Continuous):
         x = x.float().view(-1, 1)
      x = self.embed(x)
      return x

class Env(nn.Module):
   def __init__(self, config):
      super().__init__()
      h = config.HIDDEN
      self.embeddings = {}
      self.config = config

      self.emb, self.proj  = self.init(config)

   def init(self, config, name=None):
      emb  = nn.ModuleDict()
      proj = nn.ModuleDict()
      for name, subnet in config.static:
         n = 0
         emb[name] = nn.ModuleDict()
         for param, val in subnet:
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

   def forward(self, env, ent):
      stim = self.config.dynamic(env, ent)
      features, embed = {}, {}
      for name, subnet in stim.items():
         merged = self.merge(subnet)
         feats = []
         for param, val in merged.items():
            val = torch.Tensor(val)
            feats.append(self.emb[name][param](val))
         feats = torch.cat(feats, 1)
         feats = self.proj[name](feats)
         features[name] = feats
         zipped = dict(zip(subnet.keys(), feats.split(1)))
         embed = {**embed, **zipped}
      return features, embed
