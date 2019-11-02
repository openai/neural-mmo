from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from itertools import chain

from forge.blade.io import Stimulus as StaticStimulus
from forge.blade.io import Action as StaticAction
from forge.ethyr.io import Serial, utils
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

      self.action = nn.Embedding(StaticAction.n, config.HIDDEN)

   def initSubnets(self, config, name=None):
      '''Initialize embedding networks'''
      emb  = nn.ModuleDict()
      for name, subnet in StaticStimulus:
         name = '-'.join(name)
         emb[name] = nn.ModuleDict()
         for param, val in subnet:
            param = '-'.join(param)
            emb[name][param] = TaggedInput(val(config), config)
      self.emb = emb

   def actions(self, embeddings, objLookup):
      '''Embed actions'''
      embed = []
      for atn in StaticAction.actions:
         #Brackets on atn.serial?
         idx = torch.Tensor([atn.idx])
         idx = idx.long()#.to(self.config.DEVICE)

         emb = self.action(idx)
         embed.append(emb)

         key = Serial.key(atn)
         objLookup.add(key)

      padEmbed = emb * 0
      embed.append(padEmbed)
      embed.append(padEmbed)

      objLookup.add(tuple([0]*Serial.KEYLEN))
      objLookup.add(tuple([-1]*Serial.KEYLEN))
      
      embed      = torch.cat(embed)
      embeddings = torch.cat([embeddings, embed])
   
      return embeddings, objLookup

   def attrs(self, group, attn, subnet):
      '''Embed and pack attributes of each entity'''
      embeddings = []
      for param, val in subnet.items():
         param = '-'.join(param)
         val = torch.Tensor(val)#.to(self.config.DEVICE)
         emb = self.emb[group][param](val)
         embeddings.append(emb)


      embeddings = torch.stack(embeddings, -2)

      #Batch, ents, nattrs, hidden
      embeddings = attn(embeddings)
      return embeddings

   #Okay. You have some issues
   #Dimensions are batch, nEnts/tiles, nAttrs, hidden
   #So you need to go down to 3 dims for 2nd tier
   #batch, nEnts/tiles, hidden
   #And down to batch, hidden for final tier
   def forward(self, net, inputs, database, objLookup):
      #features, lookup = {}, Lookup()
      #self.actions(lookup)
 
      embeddings = []
      #Pack entities of each observation set
      for group, stim in database.items():
         embeddings.append(self.attrs(group, net.attn.emb, stim))

      embeddings = torch.cat(embeddings)
      features   = []
      for objID, inp in inputs.items():
         dat = []
         for group, stim in inp.stim.items():
            dat += [embeddings[objLookup.get(e)] for e in stim]
         dat = torch.stack(dat)
         dat = net.attn.ent(dat)
         features.append(dat)

      stims = torch.stack(features)
      stims = dict(zip(inputs.keys(), stims))

      for entID, inp in stims.items():
         inputs[entID].stim = inp

      embeddings, objLookup = self.actions(embeddings, objLookup)
      return embeddings, objLookup

