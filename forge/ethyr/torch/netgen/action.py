from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn

from forge.ethyr.torch.utils import classify
from forge.blade.io import action

class NetTree(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.config = config
      self.h = config.HIDDEN

      self.net = VariableDiscreteAction(
               self.config, self.h, self.h)

   def buffered(self, stim, embed, action):
      #Deserialize action and argument
      #Just fake a key for now
      key = 0
      outs = {}
      for atn in action:
         args, idx = atn
         idx = torch.Tensor([idx]).long()
         out, _    = self.net(stim, embed, args)#?
         outs[key] = [out[0], idx]
         key += 1

      return outs

   def standard(self, stim, embed, env, ent):
      actionTree = action.Dynamic(env, ent, action.Static, self.config)
      atn = actionTree.root

      args, done = actionTree.next(env, ent, atn)
      atnArgs = action.ActionArgs(atn, None)

      while not done:
         out, idx =  self.net(stim, embed, args)
         atn = args[int(idx)]

         args, done = actionTree.next(env, ent, atn, (args, idx))

      outs = actionTree.outs
      atn  = actionTree.atnArgs

      return atn, outs

   def forward(self, stim, embed, *args, buffered=False):
      n = 2
      atns, outs = [], {}
      if buffered: #Provide env action buffers
         for arg in args:
            out = self.buffered(stim, embed, arg) 
            outs = {**outs, **out}
      else: #Need access to env
         env, ent = args
         for _ in range(n):
            atn, out = self.standard(stim, embed, env, ent)
            atns.append(atn)
            outs = {**outs, **out}
      return atns, outs
         

class Action(nn.Module):
   def __init__(self, net, config):
      super().__init__()
      self.net = net
      self.config = config

   def forward(self, stim, embed, args, variable):
      targs = torch.cat([embed[e] for e in args])
      out, idx = self.net(stim, targs)
      return out, idx 

class ConstDiscreteAction(Action):
   def __init__(self, config, h, ydim):
      super().__init__(ConstDiscrete(h, ydim), config)

   def forward(self, stim, embed, args):
      return super().forward(stim, embed, args, variable=False)

class VariableDiscreteAction(Action):
   def __init__(self, config, xdim, h):
      super().__init__(VariableDiscrete(xdim, h), config)

   def forward(self, stim, embed, args):
      return super().forward(stim, embed, args, variable=True)

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

####### End network modules

