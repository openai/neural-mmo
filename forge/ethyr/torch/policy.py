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

class Val(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.config = config
      self.net = torch.nn.Linear(config.HIDDEN, 1)

   def forward(self, stim):
      return self.net(stim)

class Hidden(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.net = Transformer(config.HIDDEN, config.NHEAD) 
      self.val = Val(config) 

   def forward(self, stim):
      ret = self.net(stim)
      val = self.val(stim)
      return ret, val

class Net(nn.Module):
   def __init__(self, config, device, mapActions=True):
      super().__init__()
      self.config = config

      self.net = nn.ModuleList([Hidden(config) 
            for _ in range(config.NPOP)])
      self.env    = Env(config, device, mapActions)
      self.action = NetTree(config)

   def input(self, stim):
      #TODO: Need to select net index
      stim, embed = self.env(self.net[0].net, stim) 
      val                  = self.net[0].val(stim) 
      return stim, embed, val

   def forward(self, stim, *args, buffered=False): 
      #Add in action processing to input? Or maybe output embed?
      stim, embed, val = self.input(stim)
      stim = stim[0]
      atns, outs = self.action(stim, embed, *args, buffered=buffered)
      return atns, outs, val


