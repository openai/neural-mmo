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
   def __init__(self, config, project=True):
      super().__init__()
      h = config.HIDDEN
      self.embeddings = {}
      self.config = config
      self.project = project

      self.emb, self.proj1, self.proj2  = self.init(config)

   def init(self, config, name=None):
      emb  = nn.ModuleDict()
      proj1 = nn.ModuleDict()
      proj2 = nn.ModuleDict()
      for name, subnet in config.static:
         n = 0
         emb[name] = nn.ModuleDict()
         for param, val in subnet:
            n += config.EMBED
            emb[name][param] = Input(val(config), config)
         if self.project:
            proj1[name] = Transformer(config.HIDDEN, config.NHEAD)
            #proj2[name] = Transformer(config.HIDDEN, config.NHEAD)
      if self.project:
         proj2 = Transformer(config.HIDDEN, config.NHEAD)
      return emb, proj1, proj2

   #Don't need this much anymore -- built into dynamic stimulus
   def merge(self, dicts):
      ret = defaultdict(list)
      for name, d in dicts.items(): 
         for k, v in d.items():
            ret[k] += [v]
      return ret

   def forward(self, env, ent):
      stims = self.config.dynamic(env, ent, flat=True)
      features, embed = [], {}
      for group, stim in stims.items():
         names, subnet = stim
         feats = []
         for param, val in subnet.items():
            val = torch.Tensor(val)
            emb = self.emb[group][param](val)
            feats.append(emb.split(1))
         emb = np.array(feats).T.tolist()
         emb = [torch.cat(e) for e in emb]

         #feats = torch.cat([torch.cat(e) for e in feats], 1)
         feats = self.proj1[group](torch.stack(emb))
         embed = {**embed, **dict(zip(names, feats.split(1)))}
         features.append(feats)

         #shape = (1, *(feats.shape))
         #features[group] = self.proj2[group](feats.view(shape))

      
      features = torch.cat(features).unsqueeze(0)
      features = self.proj2(features)

      return features, embed
