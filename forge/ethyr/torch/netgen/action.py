from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn

from forge.ethyr.torch.policy.attention import MiniAttend
from forge.ethyr.torch.utils import classify
from forge.blade.io import action

class NetTree(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.config = config
      self.h = config.HIDDEN

      self.net = VariableDiscreteAction(
               self.config, self.h, self.h)

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
      atnArgs, outs = [], {}
      for atn in roots:
         atn, out = self.tree(
               stim, embed, env, ent, atn)
         atnArgs.append(atn)
         outs = {**outs, **out}

      return atnArgs, outs

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

   def blockEmbed(self, embed, nameMap, args):
      shape = (*(args.shape[:3]), self.h)
      ret   = torch.zeros(shape).to(self.config.DEVICE)
      for x, xx in enumerate(args):
         for y, yy in enumerate(xx):
            for z, zz in enumerate(yy):
               key = tuple(zz)
               if key == (0, 0, 0, 0, 0):
                  continue
               key = nameMap[key]
               key = embed[key]
               ret[x, y, z] = key
      return ret


   def _actions(self, stim, embed, actions):
      atnTensor, idxTensor, keyTensor, lenTensor = actions 
      lenTensor, atnLens = lenTensor
      nameMap, embed = embed

      atnTensor = torch.LongTensor(atnTensor).to(self.config.DEVICE)
      targs = embed[atnTensor]
      
      stim = stim.unsqueeze(1).unsqueeze(1)
      outs, _ = self.net(stim, targs)
      return outs

   def _arguments(self, stim, embed, actions):
      atnTensor, idxTensor, keyTensor, lenTensor = actions 
      lenTensor, atnLens = lenTensor
      nameMap, embed = embed

      targs = self.blockEmbed(embed, nameMap, atnTensor)
      
      stim = stim.unsqueeze(1).unsqueeze(1)
      outs, _ = self.net(stim, targs)
      return outs


   def buffered(self, stim, embed, atnArgs):
      actions   = self._arguments(stim, embed, atnArgs)
      return actions

   def forward(self, stims, embed, obs=None, actions=None):
      assert obs is None or actions is None
      atnArgList, outList = [], []

      #Provide final action buffers; do not need access to env
      if obs is None:
         outList = self.buffered(stims, embed, actions)

      #No buffers; need access to environment
      elif actions is None:
         for idx, ob in enumerate(obs):
            env, ent = ob
            stim = stims[idx]
            atnArgs, outs = self.leaves(stim, embed, env, ent)
            atnArgList.append(atnArgs)
            outList.append(outs)

      return atnArgList, outList

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
      #if len(x.shape) > 1:
      #   x = x.squeeze(-2)
      xIdx = classify(x)
      return x, xIdx

class AttnCat(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.attn = MiniAttend(h, flat=False)
      self.fc   = nn.Linear(h, 1)
      self.h = h

   def forward(self, key, vals):
      K, V = key, vals
      if len(K.shape) == 1:
         K = K.unsqueeze(0).unsqueeze(0).unsqueeze(0)
         V = V.unsqueeze(0).unsqueeze(0)
      
      #K = K.expand_as(V)
      #Yes, V, K. Otherwise all keys are equiv
      attn = self.attn(V, K)
      attn = self.fc(attn)
      attn = attn.squeeze(-1)
      #attn = self.attn(K, V).mean(-1)

      #attn = self.attn(K, V).mean(-1)
      attn = attn.squeeze(0).squeeze(0)
      return attn

class AttnPool(nn.Module):
   def __init__(self, xdim, h):
      super().__init__()
      self.fc = torch.nn.Linear(xdim, h)

   def forward(self, x):
      x = self.fc(x)
      x, _ = torch.max(x, 0)
      return x

####### End network modules

