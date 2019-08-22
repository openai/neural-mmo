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
   def __init__(self, stim, root, config):
      self.stim = stim.unsqueeze(0)

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

   def selectAction(self, stims, nameMap, embed):
      '''Select actions'''
      roots = Dynamic.leaves()
      #roots = [action.Attack]
      #roots = [action.Move]
 
      n = len(obs)
      atnArgs = [[] for _ in range(n)]
      outs    = [{} for _ in range(n)]
      for atn in roots:
         self.subselect(stims, nameMap, 
               embed, atnArgs, outs, atn)

      return atnArgs, outs

   def subselect(self, stims, nameMap, embed, 
         atnArgsList, outsList, root=Static):
      '''Select a single action'''

      packets = {}
      inputs = zip(stims, atnArgsList, outsList)
      for idx, packet in enumerate(inputs):
         stim, atnArgs, outs = packet
         pkt = Packet(stim, root, self.config)
         packets[idx] = pkt
         
      while len(packets) > 0:
         stimTensor, targTensor, mask = Packet.merge(
            packets.values(), self.names, embed, nameMap)
         oList, idxList = self.net(stimTensor, targTensor, mask)
         Packet.step(packets.values(), oList, idxList)
         Packet.finish(packets, atnArgsList, outsList)

   def bufferedSelect(self, stim, actions, nameMap, embed):
      atnTensor, atnTensorLens, atnLens, atnLenLens = actions

      batch, nAtn, nArgs, nAtnArg, keyDim = atnTensor.shape
      atnTensor = atnTensor.reshape(-1, keyDim)

      targs = [tuple(e) for e in atnTensor]
      names = self.names(nameMap, targs)
      targs = embed[names]
      targs = targs.view(batch, nAtn, nArgs, nAtnArg, -1)

      #Sum the atn and arg embedding to make a key dim
      targs = targs.sum(-2)
      
      #The dot prod net does not match dims.
      stim = stim.unsqueeze(1).unsqueeze(1)
      atns, atnsIdx = self.net(stim, targs, atnLens)

      if self.config.TEST:
         atns = atns.detach()

      atns = [unpack(atn, l) for atn, l in zip(atns, atnLens)]
      return atns, atnsIdx 

   def forward(self, stims, actions, embed):
      nameMap, embed = embed

      #atnArgsList, outList = self.selectAction(stims, nameMap, embed)
      outList = self.bufferedSelect(stims, actions, nameMap, embed)

      return outList

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

   #Error is coming from masking. Goes away when no mask.
   def forward(self, stim, args, lens):
      #x = (stim * args).sum(-1)
      x = self.net(stim, args)

      lens      = torch.LongTensor(lens).unsqueeze(-1)
      n, maxLen = x.shape[0], x.shape[-1]

      inds = torch.arange(maxLen).expand_as(x)
      mask = inds < lens 
      #x[1-mask] = -np.inf

      return super().forward(x, mask)



