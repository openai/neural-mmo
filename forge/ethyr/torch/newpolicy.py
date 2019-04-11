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

from forge.ethyr.torch.netgen.action import NetTree
from forge.ethyr.torch.netgen.stim import Env

class Hidden(nn.Module):
   def __init__(self, config, ydim):
      super().__init__()
      h = config.HIDDEN
      self.fc  = torch.nn.Linear(h+1800, ydim)
      #self.fc  = torch.nn.Linear(2*h, ydim)
      self.tile = torch.nn.Linear(h, 8)

   def forward(self, stim):
      tile = stim['tile']
      ent  = stim['entity'] 
      tile = self.tile(tile).view(-1)
      #tile, _ = torch.max(tile, 0)
      ent, _  = torch.max(ent, 0)
      x = torch.cat((tile, ent))
      x = self.fc(x)
      return x

class Net(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.env    = Env(config)
      self.net    = Hidden(config, config.HIDDEN)
      self.action = NetTree(config)

   def forward(self, env, ent, stim):
      stim, embed = self.env(env, ent, stim) 
      x           = self.net(stim)
      atns, outs  = self.action(env, ent, (x, embed))
      return atns, outs

class Val(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.env = Env(config)
      self.net = Hidden(config, 1)

   def forward(self, env, ent, stim):
      stim, embed = self.env(env, ent, stim) 
      val         = self.net(stim)
      return val

