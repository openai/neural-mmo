from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

from forge.ethyr.torch.policy import linear, functional

class MaxReluBlock(nn.Module):
   def __init__(self, h, layers=2):
      super().__init__()
      self.net = linear.ReluBlock(
         h, layers, postRelu=True)
 
   def forward(self, x):
      x    = self.net(x)
      x, _ = torch.max(x, dim=-2)
      return x

class DotReluBlock(nn.Module):
   def __init__(self, h, layers=2):
      super().__init__()
      self.key = linear.ReluBlock(
         h, layers, postRelu=False)

      self.val = linear.ReluBlock(
         h, layers, postRelu=False)
   
   def forward(self, k, v):
      k = self.key(k)
      v = self.val(v)
      x = functional.dot(k, v)
      return x.squeeze(-1)


class MiniAttend(nn.Module):
   def __init__(self, h, flat=True):
      super().__init__()
      self.fc1   = nn.Linear(h, h)
      self.fc2   = nn.Linear(h, h)
      self.flat = flat

   def forward(self, x, kv=None):
      if kv is not None:
         x = x * kv
   
      x = self.fc1(x)

      #New test
      x = torch.relu(x)
      x = self.fc2(x)

      if self.flat:
         #x = x.mean(-2)
         x, _ = torch.max(x, dim=-2)

      return x

class AttnCat(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.attn = MiniAttend(h, flat=False)
      self.fc   = nn.Linear(h, 1)
      self.h = h

   def forward(self, key, vals):
      K, V = key, vals
      if len(K.shape) == 1:
         K = K.unsqueeze(0).unsqueeze(0).unsqueeze(0)
         V = V.unsqueeze(0).unsqueeze(0)

      #K = K.expand_as(V)
      attn = functional.dot(K, V)

      #Yes, V, K. Otherwise all keys are equiv
      #attn = self.attn(V, K)
      #attn = self.fc(attn)
      #attn = attn.squeeze(-1)

      #attn = self.attn(K, V).mean(-1)
      attn = attn.squeeze(0).squeeze(0)
      return attn


