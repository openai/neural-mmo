from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from forge.blade.io import stimulus, action

#Pytorch embedding wrapper that subtracts the min
class Embedding(nn.Module):
   def __init__(self, var, dim):
      super().__init__()
      self.embed = torch.nn.Embedding(var.range, dim)
      self.min = var.min

   def forward(self, x):
      return self.embed(x - self.min)

#Embedding wrapper around discrete and continuous vals
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

class TaggedInput(nn.Module):
   def __init__(self, cls, config):
      super().__init__()
      self.config = config
      h           = config.EMBED

      self.embed  = Input(cls, config)
      self.tag    = torch.nn.Embedding(1, h)

      self.proj = torch.nn.Linear(2*h, h)

   def forward(self, x):
      embed = self.embed(x)

      tag = torch.LongTensor([0])
      tag = tag#.to(self.config.DEVICE)
      tag = self.tag(tag)
      tag = tag.expand_as(embed)

      x = torch.cat((embed, tag), dim=-1)
      x = self.proj(x)
      return x


