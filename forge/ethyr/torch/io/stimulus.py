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

   def actions(self, embeddings):
      '''Embed actions'''
      embed = []
      for atn in StaticAction.arguments:
         #Brackets on atn.serial?
         idx = torch.Tensor([atn.idx])
         idx = idx.long()#.to(self.config.DEVICE)

         emb = self.action(idx)
         embed.append(emb)

      padEmbed = emb * 0
      embed.append(padEmbed)

      return torch.cat([embeddings, *embed])

   def attrs(self, name, attn, entities):
      '''Embed and pack attributes of each entity'''
      embeddings = []
      for param, val in entities.attributes.items():
         param = '-'.join(param)
         val = torch.Tensor(val)#.to(self.config.DEVICE)
         emb = self.emb[name][param](val)
         embeddings.append(emb)

      embeddings = torch.stack(embeddings, -2)

      #Batch, ents, nattrs, hidden
      embeddings = attn(embeddings)
      return embeddings

   def forward(self, net, inp):
      observationTensor = []
      embeddings = []

      #Pack entities of each attribute set
      for name, entities in inp.obs.entities.items():
         embs = self.attrs(name, net.attributes[name], entities)
         embeddings.append(embs)

      #Pack entities of each observation
      entityLookup = torch.cat(embeddings)
      for objID, idxs in inp.obs.names.items():
         emb = entityLookup[[e for e in idxs]]
         obs = net.entities(emb)
         observationTensor.append(obs)

      entityLookup = self.actions(entityLookup)
      observationTensor = torch.stack(observationTensor)
      return observationTensor, entityLookup
