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

      self.net = ConstDiscreteAction(
            self.config, self.h, 4)
      #self.net = VariableDiscreteAction(
      #         self.config, self.h, self.h)

   def embed(self, embed, nameMap, args):
      return torch.stack([embed[nameMap[e]] for e in args])

   def leaves(self, stim, embed, env, ent):
      #roots = action.Dynamic.leaves()
      roots = [action.static.Move]
      return self.select(stim, embed, roots, env, ent)

   def root(self, stim, embed, env, ent):
      roots = [action.Static for _ in 
            range(self.config.NATN)]
      return self.select(stim, embed, roots, env, ent)

   def select(self, stim, embed, roots, env, ent):
      atns, outs = [], {}
      for atn in roots:
         atn, out = self.tree(
               stim, embed, env, ent, atn)
         atns.append(atn)
         outs = {**outs, **out}

      return atns, outs

   def tree(self, stim, embed, env, ent, atn=action.Static):
      actionTree = action.Dynamic(env, ent, self.config)
      args, done = actionTree.next(env, ent, atn)
      nameMap, embed = embed

      while not done:
         targs = self.embed(embed, nameMap, args)
         out, idx =  self.net(stim, targs)

         atn = args[int(idx)]
         args, done = actionTree.next(env, ent, atn, (args, idx))

      outs = actionTree.outs
      atn  = actionTree.atnArgs
      return atn, outs

   def buffered(self, stim, embed, actions):
      atnTensor, idxTensor, keyTensor, lenTensor = actions 
      lenTensor, atnLens = lenTensor
      nameMap, embed = embed

      atnTensor = torch.LongTensor(atnTensor).to(self.config.DEVICE)
      targs = embed[atnTensor]
      
      stim = stim.unsqueeze(1).unsqueeze(1)
      outs, _ = self.net(stim, targs)
      return outs

   def forward(self, stims, embed, obs=None, actions=None):
      assert obs is None or actions is None
      atnList, outList = [], []

      #Provide final action buffers; do not need access to env
      if obs is None:
         outList = self.buffered(stims, embed, actions)

      #No buffers; need access to environment
      elif actions is None:
         atnList, outList = [], []
         for idx, ob in enumerate(obs):
            env, ent = ob
            stim = stims[idx]
            atns, outs = self.leaves(stim, embed, env, ent)
            atnList.append(atns)
            outList.append(outs)

      return atnList, outList

class Action(nn.Module):
   def __init__(self, net, config):
      super().__init__()
      self.net = net
      self.config = config
      
   def forward(self, stim, targs, variable):
      out, idx = self.net(stim, targs)
      return out, idx 

class ConstDiscreteAction(Action):
   def __init__(self, config, h, ydim):
      super().__init__(ConstDiscrete(h, ydim), config)

   def forward(self, stim, args):
      return super().forward(stim, args, variable=False)

class VariableDiscreteAction(Action):
   def __init__(self, config, xdim, h):
      super().__init__(VariableDiscrete(xdim, h), config)

   def forward(self, stim, args):
      return super().forward(stim, args, variable=True)

####### Network Modules
class ConstDiscrete(nn.Module):
   def __init__(self, h, ydim):
      super().__init__()
      self.fc1 = torch.nn.Linear(h, ydim)

   def forward(self, x, _):
      x = self.fc1(x)
      if len(x.shape) > 1:
         x = x.squeeze(-2)
      xIdx = classify(x)
      return x, xIdx

class VariableDiscrete(nn.Module):
   def __init__(self, xdim, h):
      super().__init__()
      self.attn  = AttnCat(h)

   #Arguments: stim, action/argument embedding
   def forward(self, key, vals):
      x = self.attn(key, vals)
      xIdx = classify(x)
      return x, xIdx

class AttnCat(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.fc = torch.nn.Linear(2*h, 1)
      self.h = h

   def forward(self, key, vals):
      key = key.expand_as(vals)
      x = torch.cat((key, vals), dim=-1)
      x = self.fc(x).squeeze(-1)
      return x

class AttnPool(nn.Module):
   def __init__(self, xdim, h):
      super().__init__()
      self.fc = torch.nn.Linear(xdim, h)

   def forward(self, x):
      x = self.fc(x)
      x, _ = torch.max(x, 0)
      return x

####### End network modules

