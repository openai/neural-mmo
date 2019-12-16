'''Observation processing module'''

from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

from forge.blade.io import stimulus, action

class Env(nn.Module):
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
      self.device = config.DEVICE
      self.config = config
      self.h      = h

      #Assemble network modules
      self.initEmbeddings(embeddings)
      self.initAttributes(attributes)
      self.initEntities(entities)

      self.action = nn.Embedding(action.Static.n, config.HIDDEN)

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

   def actions(self, embeddings):
      '''Embed actions'''
      embed = []
      for atn in action.Static.arguments:
         idx = torch.Tensor([atn.idx])
         idx = idx.long().to(self.device)

         emb = self.action(idx)
         embed.append(emb)

      return torch.cat([embeddings, *embed])

   def attrs(self, name, attn, entities):
      '''Embed and pack attributes of each entity'''
      embeddings = []
      for param, val in entities.attributes.items():
         param = '-'.join(param)
         val = torch.Tensor(val).to(self.device)
         emb = self.emb[name][param](val)
         embeddings.append(emb)

      embeddings = torch.stack(embeddings, -2)

      #Batch, ents, nattrs, hidden
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
      observationTensor = []
      embeddings = []

      #Pack entities of each attribute set
      for name, entities in inp.obs.entities.items():
         embs = self.attrs(name, self.attributes[name], entities)
         embeddings.append(embs)

      #Pack entities of each observation
      entityLookup = torch.cat(embeddings)
      for objID, idxs in inp.obs.names.items():
         emb = entityLookup[[e for e in idxs]]
         obs = self.entities(emb)
         observationTensor.append(obs)

      entityLookup = self.actions(entityLookup)
      observationTensor = torch.stack(observationTensor)
      return observationTensor, entityLookup
