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

class Hidden(nn.Module):
   def __init__(self, config, ydim):
      super().__init__()
      self.config = config
      h = config.HIDDEN

      self.net = torch.nn.Linear(2*h, ydim)

   def forward(self, stim):
      stim = list(stim.values())
      stim = torch.cat(stim, 1)
      return self.net(stim)
      
      #tile = self.tile(tile).view(-1)
      #tile, _ = torch.max(stim['tile'], 0)
      #ent, _  = torch.max(stim['entity'], 0)
      x = torch.cat((tile, ent))
      x = self.fc(x)
      return x

class Net(nn.Module):
   def __init__(self, config, device):
      super().__init__()
      net = Transformer(config.HIDDEN, config.NHEAD)
      self.env    = Env(net, config, device)
      #self.net    = Hidden(config, config.HIDDEN)
      #self.val    = Hidden(config, 1)
      self.val    = nn.Linear(config.HIDDEN, 1)
      self.action = NetTree(config)
      self.config = config

   def forward(self, env, ent):
      stim, embed = self.env(env, ent) 
      val         = self.val(stim) 
      atns, outs  = self.action(env, ent, (stim, embed))
      return atns, outs, val
