from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from forge.blade.io import action

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
      self.config = config
      self.h = config.HIDDEN

      for atn in ActionTree.flat():
         self.add(atn)

   def add(self, cls):
      cls = self.map(cls)
      if cls is action.AttackStyle:
         self.net[cls.__name__] = AttackNet(self.config, self.h, self.net['Attack'].envNet)
      elif cls.nodeType in (NodeType.SELECTION, NodeType.CONSTANT):
         n = len(cls.edges)
         self.net[cls.__name__] = EnvConstDiscreteAction(self.config, self.h, n)
      elif cls.nodeType is NodeType.VARIABLE:
         self.net[cls.__name__] = EnvVariableDiscreteAction(self.config, self.h)

   def module(self, cls):
      cls = self.map(cls)
      return self.net[cls.__name__]

   #Force shared networks across particular nodes
   def map(self, cls):
      if cls in (action.Melee, action.Range, action.Mage):
         cls = action.AttackStyle
      return cls

   def forward(self, env, ent, stim):
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
      return atns, outs

class Action(nn.Module):
   def __init__(self, net, config):
      super().__init__()
      self.net = net
      self.config = config

   def forward(self, env, ent, action, stim, targs=None):
      if targs is None:
         atn, atnIdx = self.net(stim)
      else:
         atn, atnIdx = self.net(stim, targs)
      leaves = action.args(env, ent, self.config)
      action = leaves[int(atnIdx)]
      return action, (atn.squeeze(0), atnIdx)

class AttackNet(Action):
   def __init__(self, config, h, envNet):
      net    = VariableDiscrete(3*h, h)
      super().__init__(net, config)
      self.envNet = envNet

      entDim = 11
      self.h = h

      self.styleEmbed = torch.nn.Embedding(3, h)
      self.targEmbed  = EntEmbed(entDim, h)

   def forward(self, env, ent, action, stim):
      targs = action.args(env, ent, self.config)
      targs = self.targEmbed(targs)

      atnIdx = torch.tensor(action.index)
      atns  = self.styleEmbed(atnIdx).expand(len(targs), self.h)
      targs = torch.cat((atns, targs), 1)

      stim = self.envNet(stim.conv, stim.flat, stim.ents)
      return super().forward(env, ent, action, stim, targs)

class ConstDiscreteAction(Action):
   def __init__(self, config, h, ydim):
      super().__init__(ConstDiscrete(h, ydim), config)

class VariableDiscreteAction(Action):
   def __init__(self, config, xdim, h):
      super().__init__(VariableDiscrete(xdim, h), config)

class EnvConstDiscreteAction(ConstDiscreteAction):
   def __init__(self, config, h, ydim):
      super().__init__(config, h, ydim)
      self.envNet = Env(config)

   def forward(self, env, ent, action, stim):
      stim = self.envNet(stim.conv, stim.flat, stim.ents)
      return super().forward(env, ent, action, stim)

class EnvVariableDiscreteAction(VariableDiscreteAction):
   def __init__(self, config, xdim, h):
      super().__init__(config, xdim, h)
      self.envNet = Env(config)

   def forward(self, env, ent, action, stim, targs):
      stim = self.envNet(stim.conv, stim.flat, stim.ents)
      return super().forward(env, ent, action, stim, targs)

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
      self.attn = AttnCat(xdim, h)

   #Arguments: stim, action/argument embedding
   def forward(self, key, vals):
      x = self.attn(key, vals)
      xIdx = classify(x)
      return x, xIdx

class AttnCat(nn.Module):
   def __init__(self, xdim, h):
      super().__init__()
      #self.fc1 = torch.nn.Linear(xdim, h)
      #self.fc2 = torch.nn.Linear(h, 1)
      self.fc = torch.nn.Linear(xdim, 1)
      self.h = h

   def forward(self, x, args):
      n = args.shape[0]
      x = x.expand(n, self.h)
      xargs = torch.cat((x, args), dim=1)

      x = self.fc(xargs)
      #x = F.relu(self.fc1(xargs))
      #x = self.fc2(x)
      return x.view(1, -1)
####### End network modules

class ValNet(nn.Module):
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

class EntEmbed(nn.Module):
   def __init__(self, entDim, h):
      super().__init__()
      self.ent = torch.nn.Linear(entDim, h)

   def forward(self, ents):
      ents = torch.tensor([e.stim for e in ents]).float()
      return self.ent(ents)


class Env(nn.Module):
   def __init__(self, config):
      super().__init__()
      h = config.HIDDEN
      entDim = 11 # + 225

      self.fc1  = torch.nn.Linear(3*h, h)
      self.embed = torch.nn.Embedding(7, 7)

      self.conv = torch.nn.Linear(1800, h)
      self.flat = torch.nn.Linear(entDim, h)
      self.ents = Ent(entDim, h)

      stim = static.stim(config)

   def init(self, config, stim):
      net = nn.ModuleDict()
      T()
      
      n = 0
      for key, node in stim.items():
         #if not issubclass(node, static.Node):
         #   net[node] = self.init(config, node)

         n += node.n
         if issubclass(val, static.Discrete):
            net[key] = torch.nn.Embedding(node.val, config.HIDDEN)
      T()
      net[stim] = torch.nn.Linear(n, config.HIDDEN)
      

   def forward(self, conv, flat, ents):
      tiles, nents = conv[0], conv[1]
      nents = nents.view(-1)

      tiles = self.embed(tiles.view(-1).long()).view(-1)
      conv = torch.cat((tiles, nents))

      conv = self.conv(conv)
      ents = self.ents(ents)
      flat = self.flat(flat)

      x = torch.cat((conv, flat, ents)).view(1, -1)
      x = self.fc1(x)
      #Removed relu (easier training, lower policy cap)
      #x = torch.nn.functional.relu(self.fc1(x))
      return x
