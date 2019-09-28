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

   def actions(self, lookup):
      '''Embed actions'''
      for atn in StaticAction.actions:
         #Brackets on atn.serial?
         idx = torch.Tensor([atn.idx])
         idx = idx.long()#.to(self.config.DEVICE)

         emb = self.action(idx)
         #Dirty hack -- remove duplicates
         lookup.add([atn], [emb])

         key = Serial.key(atn)
         #What to replace serial with here
         lookup.add([key], [emb])

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
      embeddings = attn.emb(embeddings)

      #Batch, ents, hidden
      features   = attn.ent(embeddings)

      return embeddings, features

   #Okay. You have some issues
   #Dimensions are batch, nEnts/tiles, nAttrs, hidden
   #So you need to go down to 3 dims for 2nd tier
   #batch, nEnts/tiles, hidden
   #And down to batch, hidden for final tier
   def forward(self, net, stims):
      features, lookup = {}, Lookup()
      self.actions(lookup)
 
      #Pack entities of each observation set
      for group, stim in stims.items():
         names, subnet = stim
         embs, feats     = self.attrs(group, net.attns[group], subnet)
         features[group] = feats

         #Unpack and flatten for embedding
         lens = [len(e) for e in names]
         vals = utils.unpack(embs, lens, dim=1)
         for k, v in zip(names, vals):
            v = v.split(1, dim=0)
            lookup.add(k, v)


      k = [tuple([0]*Serial.KEYLEN)]
      v = [v[-1] * 0]
      lookup.add(k, v)

      k = [tuple([-1]*Serial.KEYLEN)]
      v = [v[-1] * 0]
      lookup.add(k, v)
   
      #Concat feature block
      #me = features['Entity'][:, 0, :]
      #features = torch.cat(list(features.values()), dim=-2)
      #features = net.attns['Meta'](me, features)
      
      feats = features 
      features = list(features.values())
      features = torch.stack(features, -2)

      #Batch, group (tile/ent), hidden
      features = net.attns['Meta'](features)#.squeeze(0)
      
      embed = lookup.table()
      #embed = None
      return features, embed

