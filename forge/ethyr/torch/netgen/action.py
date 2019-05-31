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
      atnTensor, idxTensor, lenTensor = action
      atnTensor, atnLens = atnTensor
      nameMap, embed = embed
      #idxTensor, idxLens = idxTensor
      #lenTensor, lenLens = lenTensor

      #Deserialize action and argument
      #Just fake a key for now
      key = 0
      outs = {}
      #for atn in action:
      atnTensor = torch.LongTensor(atnTensor).to('cuda:0')
      
      #for emb, atn in zip(embed, atnTensor):
      #   pass
      #Embed of atn

      #Zeroed out embed for bounds. This is an issue.
      targs = embed[atnTensor]
      #targs = torch.stack([emb[atn] for emb, atn in zip(embed, atnTensor)])
      #[targ[:, idx] for idx in range(len(targs))]
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

   def forward(self, stim, embed, *args, buffered=False):
      n = 2
      atns, outs = [], {}
      if buffered: #Provide env action buffers
         outs = self.buffered(stim, embed, *args) 
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

