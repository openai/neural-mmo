from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from forge.blade.io import stimulus, action
from forge.ethyr.torch.netgen.stim import Env
from forge.ethyr.torch.netgen.action import NetTree

from forge.ethyr.torch.modules import Transformer
from forge.ethyr.torch.modules.transformer import MiniAttend, ScaledDotProductAttention

class Val(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.config = config
      self.net = torch.nn.Linear(config.HIDDEN, 1)

   def forward(self, stim):
      return self.net(stim)

class Net(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.net = Transformer(config.HIDDEN, config.NHEAD, nLayers=1) 
      self.val = Val(config) 

   def forward(self, stim):
      ret = self.net(stim)
      val = self.val(stim)
      return ret, val

class FCNet(nn.Module):
   def __init__(self, config):
      super().__init__()
      h = config.HIDDEN
      self.flat   = nn.Linear(10*h, h)
      self.attn1 = MiniAttend(h, config.NHEAD, nLayers=1) 
      self.attn2 = MiniAttend(h, config.NHEAD, nLayers=1) 

      self.scaled = ScaledDotProductAttention(h)
      self.fc1    = nn.Linear(h, h)
      self.fc2    = nn.Linear(h, h)
      self.fc3    = nn.Linear(h, h)
 
class Net(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.net  = FCNet(config)
      self.val  = Val(config) 

   def forward(self, stim):
      ret = self.net(stim)
      val = self.val(stim)
      return ret, val

