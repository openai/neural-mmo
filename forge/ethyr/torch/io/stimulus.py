'''Observation processing module'''

from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

from forge.blade.io import stimulus, action
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.stimulus import node
from forge.blade.io.action.static import Fixed

from ray.rllib.models.repeated_values import RepeatedValues

class Input(nn.Module):
   def __init__(self, config, embeddings, attributes, entities):
      '''Network responsible for processing observations

      Args:
         config     : A configuration object
         embeddings : An attribute embedding module
         attributes : An attribute attention module
         entities   : An entity attention module
      '''
      super().__init__()
      h           = config.HIDDEN
      self.config = config
      self.h      = h

      #Assemble network modules
      self.initEmbeddings(embeddings)
      self.initAttributes(attributes)
      self.initEntities(entities)

      #self.action = nn.Embedding(action.Static.n, config.HIDDEN)

   def initEmbeddings(self, embedF):
      '''Initialize embedding networks'''
      emb  = nn.ModuleDict()
      for name, subnet in stimulus.Static:
         name = '-'.join(name)
         emb[name] = nn.ModuleDict()
         for param, val in subnet:
            param = '-'.join(param)
            emb[name][param] = embedF(val(self.config), self.config)
      self.emb = emb

   def initAttributes(self, attrF):
      '''Initialize attribute networks'''
      self.attributes = nn.ModuleDict()
      for name, subnet in stimulus.Static:  
         self.attributes['-'.join(name)] = attrF(self.config)

   def initEntities(self, entF):
      '''Initialize entity network'''
      self.entities = entF(self.config) 

   def actions(self):
      '''Embed actions'''
      embed = []
      for atn in action.Static.arguments:
         idx = torch.Tensor([atn.idx])
         idx = idx.long().to(self.device)

         emb = self.action(idx)
         embed.append(emb)

      return torch.cat(embed)

   def attrs(self, name, attn, entities):
      '''Embed and pack attributes of each entity'''
      embeddings = []
      attrs = Stimulus.dict()[name]
      #Slow probably
      for param, val in attrs:
         if type(entities) == RepeatedValues:
            val    = entities.values[param].squeeze(-1)
            embNet = self.emb[name]['-'.join(param)]
            emb    = embNet(val) 
         else:
            val = [e[param].squeeze(-1) for e in entities]
            val = torch.stack(val, 1)
            emb = self.emb[name]['-'.join(param)](val)

         embeddings.append(emb)

      #Construct: Batch, ents, nattrs, hidden
      embeddings = torch.stack(embeddings, -2)

      #Construct: Batch, ents, hidden
      embeddings = attn(embeddings)

      return embeddings

   def forward(self, inp):
      '''Produces tensor representations from an IO object

      Args:                                                                   
         inp: An IO object specifying observations                      
         

      Returns:
         observationTensor : A fixed size observation representation
         entityLookup      : A fixed size representation of each entity
      ''' 

      #Pack entities of each attribute set
      embeddings, entityLookup = [], {}
      for name, entities in inp.items():
         name = name[0] #Temp hack
         embs = self.attrs(name, self.attributes[name], entities)
         entityLookup[name] = embs
         embeddings.append(embs)

      #entityLookup[Fixed.__name__] = self.actions()

      #Pack entities of each observation
      embeddings   = torch.cat(embeddings, dim=-2)
      obsTensor    = self.entities(embeddings)

      return obsTensor, entityLookup
