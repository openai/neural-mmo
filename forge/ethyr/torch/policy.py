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
   def __init__(self, config, device):
      super().__init__()
      self.config = config

      self.net = nn.ModuleList([Hidden(config) 
            for _ in range(config.NPOP)])
      self.env    = Env(config, device)
      self.action = NetTree(config)

   def forward(self, obs):
      #TODO: Need to select net index
      stim, embed = self.env(self.net[0].net, obs) 
      val         = self.net[0].val(stim) 

      rets = []
      for ob, s in zip(obs, stim):
         env, ent = ob
         atns, outs  = self.action(env, ent, (s, embed))
         rets.append((atns, outs))

      atns, outs = zip(*rets)
      return atns, outs, val
