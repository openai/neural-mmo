from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from forge.blade.io.action import static
from forge.blade.io.action.static import ActionRoot, NodeType
from forge.blade.io.action.dynamic import ActionTree

from forge.blade.io.stim import node
from forge.ethyr.torch.netgen.action import NetTree
from forge.ethyr.torch.netgen.stim import Env

class SelfAttention(nn.Module):
   def __init__(self, xdim, h):
      super().__init__()
      self.Q = nn.Linear(xdim, h)
      self.K = nn.Linear(xdim, h)
      self.V = nn.Linear(xdim, h)
      self.scale = np.sqrt(h)
      self.h = h

   def forward(self, x):
      Q = self.Q(x)
      K = self.K(x)
      V = self.V(x)

      Kt = torch.transpose(K, 0, 1)
      QK = torch.matmul(Q, Kt)
      QK = torch.softmax(QK / self.scale, 0)
      QKV = torch.matmul(QK, V)
      return torch.mean(QKV, 0)

class Hidden(nn.Module):
   def __init__(self, config, ydim):
      super().__init__()
      self.config = config
      h = config.HIDDEN

      self.tileAttn = SelfAttention(h, h)
      self.entAttn  = SelfAttention(h, h)
      self.fc = torch.nn.Linear(2*h, ydim)

   def forward(self, stim):
      tile = self.tileAttn(stim['Tile'])
      ent  = self.entAttn(stim['Entity'])
      
      #tile = self.tile(tile).view(-1)
      #tile, _ = torch.max(stim['tile'], 0)
      #ent, _  = torch.max(stim['entity'], 0)
      x = torch.cat((tile, ent))
      x = self.fc(x)
      return x

class Net(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.env    = Env(config)
      self.net    = Hidden(config, config.HIDDEN)
      self.val    = Hidden(config, 1)
      self.action = NetTree(config)
      self.config = config

   def forward(self, env, ent):
      stim, embed = self.env(env, ent) 
      x           = self.net(stim)
      val         = self.val(stim) 
      atns, outs  = self.action(env, ent, (x, embed))
      return atns, outs, val
