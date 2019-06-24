from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from itertools import chain

from forge.blade.io import stimulus, action
from forge.blade.io import utils
from forge.blade.io.serial import Serial

from forge.ethyr.torch.policy.embed import Input, TaggedInput, Embedding

class Lookup:
   '''Lookup utility for indexing 
   (name, data) pairs'''
   def __init__(self):
      self.names = []
      self.data  = []

   def add(self, names, data):
      '''Add entries to the table'''
      self.names += names
      self.data  += data

   def table(self):
      names, data = self.names, self.data
      dat = dict(zip(names, data))
      names = set(self.names)

      idxs, data = {}, []
      for idx, name in enumerate(names):
         idxs[name] = idx
         data.append(dat[name])

      data = torch.cat(data)
      assert len(idxs) == len(data)
      return idxs, data
 
class Env(nn.Module):
   '''Network responsible for processing observations

   Args:
      config: A Config object
   '''
   def __init__(self, config):
      super().__init__()
      self.config = config
      h = config.HIDDEN
      self.h = h

      self.initSubnets(config)
      self.action = nn.Embedding(action.Static.n, self.h)

   def initSubnets(self, config, name=None):
      '''Initialize embedding networks'''
      emb  = nn.ModuleDict()
      for name, subnet in config.static:
         emb[name] = nn.ModuleDict()
         for param, val in subnet:
            emb[name][param] = TaggedInput(val(config), config)
      self.emb = emb

   def actions(self, lookup):
      '''Embed actions'''
      for atn in action.Static.actions:
         #Brackets on atn.serial?
         idx = torch.Tensor([atn.idx])
         idx = idx.long().to(self.config.DEVICE)

         emb = self.action(idx)
         #Dirty hack -- remove duplicates
         lookup.add([atn], [emb])

         key = Serial.key(atn, tuple([]))
         #What to replace serial with here
         lookup.add([key], [emb])

   def attrs(self, group, net, subnet):
      '''Embed and pack attributes of each entity'''
      feats = []
      for param, val in subnet.items():
         val = torch.Tensor(val).to(self.config.DEVICE)
         emb = self.emb[group][param](val)
         feats.append(emb)

      emb = torch.stack(feats, -2)
      emb = net.attn1(emb)

      return emb

   def forward(self, net, stims):
      features, lookup = {}, Lookup()
      self.actions(lookup)

      #Pack entities of each observation set
      for group, stim in stims.items():
         names, subnet = stim
         emb = self.attrs(group, net, subnet)
         features[group] = emb.unsqueeze(0)

         #Unpack and flatten for embedding
         lens = [len(e) for e in names]
         vals = utils.unpack(emb, lens, dim=1)
         for k, v in zip(names, vals):
            v = v.split(1, dim=0)
            lookup.add(k, v)

      #Concat feature block
      features = list(features.values())
      features = torch.cat(features, -2)
      features = net.attn2(features).squeeze(0)
      
      embed = lookup.table()
      return features, embed

