from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn

from forge.ethyr.torch.policy import attention
from forge.ethyr.torch.policy import functional
from forge.blade.io import Action as Static
from forge.blade.io.action import static as action

from forge.ethyr.io import Action as Dynamic
from forge.ethyr.io.utils import pack, unpack

from forge.blade.entity.player import Player

class Packet:
   '''Manager class for packets of actions'''
   def __init__(self, stim, ob, root, config):
      self.stim = stim.unsqueeze(0)
      self.env, self.ent = ob 

      self.tree = Dynamic(self.env, self.ent, config)
      self.args, self.done = self.tree.next(self.env, self.ent, root)

   def merge(packets, nameFunc, embed, nameMap):
      names, mask = [], []
      for p in packets:
         args =  [e.injectedSerial if hasattr(
            e, 'injectedSerial') else e for e in p.args]
         name = nameFunc(nameMap, args)
         names.append(name) 
         mask.append(len(args))
            
      names, _ = pack(names)
      names    = torch.LongTensor(names)
      targs = embed[names]

      stims = [p.stim for p in packets]
      stims = torch.stack(stims)
      return stims, targs, mask

   def step(packets, outsList, idxList):
      for p, out, idx in zip(
            packets, outsList, idxList): 

         atn  = p.args[int(idx)]
         p.args, p.done = p.tree.next(
            p.env, p.ent, atn, (p.args, idx))

   def finish(packets, atnArgsList, outsList):
      delIdx = []
      for idx, p in packets.items():
         if not p.done:
            continue
         done = False

         atnArgsList[idx].append(p.tree.atnArgs)
         outsList[idx] = {**outsList[idx], **p.tree.outs}
         delIdx.append(idx)
      
      for idx in delIdx:
         del packets[idx]


class NetTree(nn.Module):
   '''Network responsible for selecting actions

   Args:
      config: A Config object
   '''
   def __init__(self, config):
      super().__init__()
      self.config = config
      self.h = config.HIDDEN

      self.net = VariableDiscreteAction(
               self.config, self.h, self.h)

   def names(self, nameMap, args):
      return np.array([nameMap[e] for e in args])

   def selectAction(self, stims, obs, nameMap, embed):
      '''Select actions'''
      #roots = Dynamic.leaves()
      #roots = [action.Attack]
      roots = [action.Move]
 
      n = len(obs)
      atnArgs = [[] for _ in range(n)]
      outs    = [{} for _ in range(n)]
      for atn in roots:
         self.subselect(stims, obs, nameMap, 
               embed, atnArgs, outs, atn)

      return atnArgs, outs

   def subselect(self, stims, obs, nameMap, embed, 
         atnArgsList, outsList, root=Static):
      '''Select a single action'''

      packets = {}
      inputs = zip(stims, obs, atnArgsList, outsList)
      for idx, packet in enumerate(inputs):
         stim, ob, atnArgs, outs = packet
         pkt = Packet(stim, ob, root, self.config)
         packets[idx] = pkt
         
      while len(packets) > 0:
         stimTensor, targTensor, mask = Packet.merge(
            packets.values(), self.names, embed, nameMap)
         oList, idxList = self.net(stimTensor, targTensor, mask)
         Packet.step(packets.values(), oList, idxList)
         Packet.finish(packets, atnArgsList, outsList)

   def bufferedSelect(self, stim, nameMap, embed, actions):
      atnTensor, idxTensor, keyTensor, lenTensor = actions 
      lenTensor, atnLens = lenTensor

      i, j, k, n = atnTensor.shape
      atnTensor = atnTensor.reshape(-1, n)
      targs = [tuple(e) for e in atnTensor]
      names = self.names(nameMap, targs)
      targs = embed[names]
      targs = targs.view(i, j, k, -1)
      
      #The dot prod net does not match dims.
      stim = stim.unsqueeze(1).unsqueeze(1)
      outs, _ = self.net(stim, targs, lenTensor)
      return None, outs

   def forward(self, stims, embed, obs=None, actions=None):
      assert obs is None or actions is None
      nameMap, embed = embed

      #Provide final action buffers; do not need access to env
      if obs is None:
         atnArgsList, outList = self.bufferedSelect(stims, nameMap, embed, actions)

      #No buffers; need access to environment
      elif actions is None:
         atnArgsList, outList = self.selectAction(stims, obs, nameMap, embed)

      return atnArgsList, outList

class Action(nn.Module):
   '''Head for selecting an action'''
   def forward(self, x, mask=None):
      xIdx = functional.classify(x, mask)
      return x, xIdx

class ConstDiscreteAction(Action):
   '''Head for making a discrete selection from
   a constant number of candidate actions'''
   def __init__(self, config, h, ydim):
      super().__init__()
      self.net = torch.nn.Linear(h, ydim)

   def forward(self, stim):
      x = self.net(stim)
      if len(x.shape) > 1:
         x = x.squeeze(-2)
      return super().forward(x)

class VariableDiscreteAction(Action):
   '''Head for making a discrete selection from
   a variable number of candidate actions'''
   def __init__(self, config, xdim, h):
      super().__init__()
      #self.net = attention.AttnCat(h)
      #self.net = attention.BareMetal(h)
      self.net = attention.DotReluBlock(h)
      #self.net = functional.dot
      #self.net = torch.nn.Linear(h, 4)

   def forward(self, stim, args, lens):
      x = (stim * args).sum(-1)
      #x = self.net(stim, args)

      lens      = torch.LongTensor(lens).unsqueeze(-1)
      n, maxLen = x.shape[0], x.shape[-1]

      inds = torch.arange(maxLen).expand_as(x)
      mask = inds < lens 
      #x[1-mask] = -np.inf

      return super().forward(x, mask)



