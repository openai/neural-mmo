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

class Packet:
   '''Manager class for packets of actions'''
   def __init__(self, stim, ob, root, config):
      self.stim = stim.unsqueeze(0)
      self.env, self.ent = ob 

      self.tree = Dynamic(self.env, self.ent, config)
      self.args, self.done = self.tree.next(self.env, self.ent, root)

   def merge(packets, nameFunc, embed, nameMap):
      names = [nameFunc(nameMap, p.args) for p in packets]
      names = torch.LongTensor(np.stack(names))
      targs = embed[names]

      stims = [p.stim for p in packets]
      stims = torch.stack(stims)

      return stims, targs

   def step(packets, outsList, idxList):
      for p, out, idx in zip(
            packets.values(), outsList, idxList): 

         atn  = p.args[int(idx)]
         p.args, p.done = p.tree.next(
            p.env, p.ent, atn, (p.args, idx))
      
   def finish(packets, atnArgsList, outsList):
      delIdx = []
      for idx, p in packets.items():
         if not p.done:
            continue

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

      #self.net = ConstDiscreteAction(
      #         self.config, self.h, 4)


   def names(self, nameMap, args):
      return np.array([nameMap[e] for e in args])
      #return torch.stack([embed[nameMap[e]] for e in args])

   def leaves(self, stims, obs, embed):
      #roots = action.Dynamic.leaves()
      roots = [action.Move]
      #roots = [action.Static for _ in 
      #      range(self.config.NATN)]
 
      return self.select(stims, obs, embed, roots)

   def select(self, stims, obs, embed, roots):
      '''Select actions'''
      #atnArgs, outs = [], {}
      n = len(obs)
      atnArgs = [[] for _ in range(n)]
      outs    = [{} for _ in range(n)]
      for atn in roots:
         self.tree(stims, obs, 
            embed, atnArgs, outs, atn)

      return atnArgs, outs


   def tree(self, stims, obs, embed, 
         atnArgsList, outsList, root=Static):
      '''Select a single action'''
      nameMap, embed = embed

      packets = {}
      inputs = zip(stims, obs, atnArgsList, outsList)
      for pkt, packet in enumerate(inputs):
         stim, ob, atnArgs, outs = packet
         packets[pkt] = Packet(stim, ob, root, self.config)
         
      while len(packets) > 0:
         stimTensor, targTensor = Packet.merge(
            packets.values(), self.names, embed, nameMap)
         oList, idxList = self.net(stimTensor, targTensor)
         Packet.step(packets, oList, idxList)
         Packet.finish(packets, atnArgsList, outsList)
      return

         
      '''
         targs = self

         stim, ob, atnArgs, outs = packet
         stim = stim.unsqueeze(0)
         env, ent = ob

         actionTree = action.Dynamic(env, ent, self.config)
         args, done = actionTree.next(env, ent, root)
         
         while not done:
            targs = self.embed(embed, nameMap, args)
            out, idx =  self.net(stim, targs)

            atn = args[int(idx)]
            args, done = actionTree.next(env, ent, atn, (args, idx))

         atnArgsList[pkt].append(actionTree.atnArgs)
         outsList[pkt] = {**outsList[pkt], **actionTree.outs}
      '''

   def buffered(self, stim, embed, actions):
      atnTensor, idxTensor, keyTensor, lenTensor = actions 
      lenTensor, atnLens = lenTensor
      nameMap, embed = embed

      i, j, k, n = atnTensor.shape
      atnTensor = atnTensor.reshape(-1, n)
      targs = [tuple(e) for e in atnTensor]
      names = self.names(nameMap, targs)
      targs = embed[names]
      targs = targs.view(i, j, k, -1)
      
      stim = stim.unsqueeze(1).unsqueeze(1)
      outs, _ = self.net(stim, targs)
      return None, outs

   def forward(self, stims, embed, obs=None, actions=None):
      assert obs is None or actions is None

      #Provide final action buffers; do not need access to env
      if obs is None:
         atnArgsList, outList = self.buffered(stims, embed, actions)

      #No buffers; need access to environment
      elif actions is None:
         atnArgsList, outList = self.leaves(stims, obs, embed)

      return atnArgsList, outList

class Action(nn.Module):
   '''Head for selecting an action'''
   def forward(self, x):
      xIdx = functional.classify(x)
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

   def forward(self, stim, args):
      #x = functional.dot(stim, args)
      x = self.net(stim, args)
      #x = x.squeeze(-2)

      return super().forward(x)



