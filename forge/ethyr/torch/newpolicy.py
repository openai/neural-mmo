from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from forge.blade.action.tree import ActionTree
from forge.blade.action import action
from forge.blade.action.action import ActionRoot, NodeType

def classify(logits):
   if len(logits.shape) == 1:
      logits = logits.view(1, -1)
   distribution = Categorical(1e-3+F.softmax(logits, dim=1))
   atn = distribution.sample()
   return atn

class NetTree(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.net = nn.ModuleDict()
      self.env = Env(config)
      self.val = Val(config)

      self.config = config
      self.h = config.HIDDEN

      for atn in ActionTree.flat(ActionRoot):
         self.add(atn)

   def add(self, cls):
      if cls.nodeType in (NodeType.SELECTION, NodeType.CONSTANT):
         n = len(cls.edges)
         self.net[cls.__name__] = ConstDiscreteAction(self.config, self.h, n)
      elif cls.nodeType is NodeType.VARIABLE:
         self.net[cls.__name__] = VariableDiscreteAction(self.config, self.env.entDim, self.h)

   def module(self, cls):
      return self.net[cls.__name__]

   def forward(self, env, ent, stim):
      val = self.val(stim.conv, stim.flat, stim.ents)
      stim = self.env(stim.conv, stim.flat, stim.ents)

      actionTree = ActionTree(env, ent, ActionRoot)
      atn = actionTree.next(actionTree.root)
      while atn is not None:
         module = self.module(atn)

         assert atn.nodeType in (
               NodeType.SELECTION, NodeType.CONSTANT, NodeType.VARIABLE)

         if atn.nodeType in (NodeType.SELECTION, NodeType.CONSTANT):
            nxtAtn, outs = module(env, ent, atn, stim)
            atn = actionTree.next(nxtAtn, outs=outs)
         elif atn.nodeType is NodeType.VARIABLE:
            argument, outs = module(env, ent, atn, stim)
            atn = actionTree.next(atn, argument, outs)

      atns, outs = actionTree.unpackActions()
      return atns, outs, val

class Action(nn.Module):
   def __init__(self, net, config):
      super().__init__()
      self.net = net
      self.config = config

   def forward(self, env, ent, action, stim, variable):
      leaves = action.args(env, ent, self.config)
      if variable:
         targs = torch.tensor([e.stim for e in leaves]).float()
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
      self.embed = nn.Linear(xdim, h) 

   #Arguments: stim, action/argument embedding
   def forward(self, key, vals):
      vals = self.embed(vals)
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
####### End network modules

class Val(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.fc = torch.nn.Linear(config.HIDDEN, 1)
      self.envNet = Env(config)

   def forward(self, conv, flat, ent):
      stim = self.envNet(conv, flat, ent)
      x = self.fc(stim)
      x = x.view(-1)
      return x

class Ent(nn.Module):
   def __init__(self, entDim, h):
      super().__init__()
      self.ent = torch.nn.Linear(entDim, h)

   def forward(self, ents):
      ents = self.ent(ents)
      ents, _ = torch.max(ents, 0)
      return ents

class Env(nn.Module):
   def __init__(self, config):
      super().__init__()
      h = config.HIDDEN
      self.entDim = 11

      self.fc1  = torch.nn.Linear(3*h, h)
      self.embed = torch.nn.Embedding(7, 7)

      self.conv = torch.nn.Linear(1800, h)
      self.flat = torch.nn.Linear(self.entDim, h)
      self.ent = Ent(self.entDim, h)

   def forward(self, conv, flat, ents):
      tiles, nents = conv[0], conv[1]
      nents = nents.view(-1)

      tiles = self.embed(tiles.view(-1).long()).view(-1)
      conv = torch.cat((tiles, nents))

      conv = self.conv(conv)
      flat = self.flat(flat)
      ents = self.ent(ents)

      x = torch.cat((conv, flat, ents)).view(1, -1)
      x = self.fc1(x)
      #Removed relu (easier training, lower policy cap)
      #x = torch.nn.functional.relu(self.fc1(x))
      return x
