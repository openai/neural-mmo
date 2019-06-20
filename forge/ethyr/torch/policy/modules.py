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
      tag = tag.to(self.config.DEVICE)
      tag = self.tag(tag)
      tag = tag.expand_as(embed)

      x = torch.cat((embed, tag), dim=-1)
      x = self.proj(x)
      return x

#Same padded (odd k)
def Conv2d(fIn, fOut, k, stride=1):
   pad = int((k-1)/2)
   return torch.nn.Conv2d(fIn, fOut, k, stride=stride, padding=pad)

def Pool(k, stride=1, pad=0):
   #pad = int((k-1)/2)
   return torch.nn.MaxPool2d(k, stride=stride, padding=pad)

def Relu():
   return torch.nn.ReLU()

class FCRelu(nn.Module):
   def __init__(self, xdim, ydim):
      super().__init__()
      self.fc = torch.nn.Linear(xdim, ydim)
      self.relu = Relu()

   def forward(self, x):
      x = self.fc(x)
      x = self.relu(x)
      return x

class ConvReluPool(nn.Module):
   def __init__(self, fIn, fOut, k, stride=1, pool=2):
      super().__init__()
      self.conv = Conv2d(fIn, fOut, k, stride)
      self.relu = Relu()
      self.pool = Pool(k)

   def forward(self, x):
      x = self.conv(x)
      x = self.relu(x)
      x = self.pool(x)
      return x
