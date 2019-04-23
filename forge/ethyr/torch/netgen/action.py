from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from forge.blade.io import action

class NetTree(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.net = nn.ModuleDict()

      self.config = config
      self.h = config.HIDDEN

      for atn in action.Dynamic.flat(action.Static):
         self.add(atn)

   def add(self, cls):
      if cls.nodeType in (action.NodeType.SELECTION, action.NodeType.CONSTANT):
         n = len(cls.edges)
         self.net[cls.__name__] = ConstDiscreteAction(self.config, self.h, n)
      elif cls.nodeType is action.NodeType.VARIABLE:
         self.net[cls.__name__] = VariableDiscreteAction(self.config, self.h, self.h)

   def module(self, cls):
      return self.net[cls.__name__]

   def forward(self, env, ent, stim):
      actionTree = action.Dynamic(env, ent, action.Static)
      atn = actionTree.next(actionTree.root)
      while atn is not None:
         module = self.module(atn)

         assert atn.nodeType in (
               action.NodeType.SELECTION, action.NodeType.CONSTANT, action.NodeType.VARIABLE)

         if atn.nodeType in (action.NodeType.SELECTION, action.NodeType.CONSTANT):
            nxtAtn, outs = module(env, ent, atn, stim)
            atn = actionTree.next(nxtAtn, outs=outs)
         elif atn.nodeType is action.NodeType.VARIABLE:
            argument, outs = module(env, ent, atn, stim)
            atn = actionTree.next(atn, argument, outs)

      atns, outs = actionTree.unpackActions()
      return atns, outs

class Action(nn.Module):
   def __init__(self, net, config):
      super().__init__()
      self.net = net
      self.config = config

   def forward(self, env, ent, action, stim, variable):
      stim, embed = stim
      leaves = action.args(env, ent, self.config)
      if variable:
         targs = torch.cat([embed[e] for e in leaves])
         atn, atnIdx = self.net(stim, targs)
      else:
         atn, atnIdx = self.net(stim)
      action = leaves[int(atnIdx)]
      return action, (atn.squeeze(0), atnIdx)

class ConstDiscreteAction(Action):
   def __init__(self, config, h, ydim):
      super().__init__(ConstDiscrete(h, ydim), config)

   def forward(self, env, ent, action, stim):
      return super().forward(env, ent, action, stim, variable=False)

class VariableDiscreteAction(Action):
   def __init__(self, config, xdim, h):
      super().__init__(VariableDiscrete(xdim, h), config)

   def forward(self, env, ent, action, stim):
      return super().forward(env, ent, action, stim, variable=True)

####### Network Modules
class ConstDiscrete(nn.Module):
   def __init__(self, h, ydim):
      super().__init__()
      self.fc1 = torch.nn.Linear(h, ydim)

   def forward(self, x):
      x = self.fc1(x)
      xIdx = classify(x)
      return x, xIdx

class VariableDiscrete(nn.Module):
   def __init__(self, xdim, h):
      super().__init__()
      self.attn  = AttnCat(h)
      #self.embed = nn.Linear(xdim, h) 

   #Arguments: stim, action/argument embedding
   def forward(self, key, vals):
      #vals = self.embed(vals)
      x = self.attn(key, vals)
      xIdx = classify(x)
      return x, xIdx

class AttnCat(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.fc = torch.nn.Linear(2*h, 1)
      self.h = h

   def forward(self, key, vals):
      key = key.expand(len(vals), self.h)
      x = torch.cat((key, vals), dim=1)
      x = self.fc(x)
      return x.view(1, -1)

class AttnPool(nn.Module):
   def __init__(self, xdim, h):
      super().__init__()
      self.fc = torch.nn.Linear(xdim, h)

   def forward(self, x):
      x = self.fc(x)
      x, _ = torch.max(x, 0)
      return x

def classify(logits):
   if len(logits.shape) == 1:
      logits = logits.view(1, -1)
   distribution = Categorical(1e-3+F.softmax(logits, dim=1))
   atn = distribution.sample()
   return atn
####### End network modules

