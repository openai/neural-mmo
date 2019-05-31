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

   def buffered(self, stim, embed, actions):
      atnTensor, idxTensor, lenTensor = actions 
      lenTensor, atnLens = lenTensor
      nameMap, embed = embed

      atnTensor = torch.LongTensor(atnTensor).to(self.config.DEVICE)
      
      targs = embed[atnTensor]
      stim = stim.unsqueeze(1).unsqueeze(1)
      outs, _ = self.net(stim, targs)
      return outs

   def standard(self, stim, embed, env, ent):
      actionTree = action.Dynamic(env, ent, action.Static, self.config)
      atn = actionTree.root

      args, done = actionTree.next(env, ent, atn)
      atnArgs = action.ActionArgs(atn, None)

      nameMap, embed = embed

      while not done:
         targs = [nameMap[e] for e in args]
         targs = [embed[targ] for targ in targs]
         targs = torch.stack(targs)
      
         out, idx =  self.net(stim, targs)
         atn = args[int(idx)]

         args, done = actionTree.next(env, ent, atn, (args, idx))

      outs = actionTree.outs
      atn  = actionTree.atnArgs

      return atn, outs

   def forward(self, stims, embed, obs=None, actions=None):
      n = 2
      assert obs is None or actions is None
      outList, atnList = [], []
      #Provide final action buffers; do not need access to env
      if obs is None:
         outList = self.buffered(stims, embed, actions)
      #No buffers; need access to environment
      elif actions is None:
         for idx, ob in enumerate(obs):
            env, ent = ob
            stim = stims[idx]
            atns, outs = [], {}
            for _ in range(n):
               atn, out = self.standard(stim, embed, env, ent)
               atns.append(atn)
               outs = {**outs, **out}
            outList.append(outs)
            atnList.append(atns)
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
      #key = key.expand(len(vals), self.h)
      key = key.expand_as(vals)
      #x = torch.cat((key, vals), dim=1)
      x = torch.cat((key, vals), dim=-1)
      x = self.fc(x).squeeze(-1)
      return x
      #return x.view(1, -1)

class AttnPool(nn.Module):
   def __init__(self, xdim, h):
      super().__init__()
      self.fc = torch.nn.Linear(xdim, h)

   def forward(self, x):
      x = self.fc(x)
      x, _ = torch.max(x, 0)
      return x

####### End network modules

