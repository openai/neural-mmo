'''Observation processing module'''

from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

from forge.blade.io import stimulus, action
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.stimulus import node

class Input(nn.Module):
   def __init__(self, config, embeddings, attributes):
      '''Network responsible for processing observations

      Args:
         config     : A configuration object
         embeddings : An attribute embedding module
         attributes : An attribute attention module
         entities   : An entity attention module
      '''
      super().__init__()
      self.embed  = config.EMBED
      self.hidden = config.HIDDEN
      self.config = config

      #Assemble network modules
      self.initSubnets(embeddings, attributes)

   def initSubnets(self, embedF, attrF):
      '''Initialize embedding and attribute networks'''
      embeddings = nn.ModuleDict()
      attributes = nn.ModuleDict()

      for _, entity in stimulus.Static:
         attributes[entity.__name__] = attrF(self.embed, self.hidden)
         embeddings[entity.__name__] = nn.ModuleDict()

         for _, attr in entity:
            val = attr(self.config)
            emb = embedF(val, self.config)
            embeddings[entity.__name__][attr.__name__] = emb

      self.embeddings = embeddings
      self.attributes = attributes

   def attrs(self, entName, attn, entities):
      '''Embed and pack attributes of each entity'''
      attrs      = Stimulus.dict()[entName]
      embeddings = []

      #Slow probably
      for attrName, attr in attrs:
         val    = entities.values[attr].squeeze(-1)
         embNet = self.embeddings[entName][attrName[-1]]
         emb    = embNet(val) 
         embeddings.append(emb)

      #Construct: Batch, ents, nattrs, hidden
      embeddings = torch.stack(embeddings, -2)

      #Construct: Batch, ents, hidden
      embeddings, scores = attn(embeddings)

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
      entityLookup = {}
      for cls, entities in inp.items():
         name = cls.__name__
         embs = self.attrs(name, self.attributes[name], entities)
         entityLookup[cls] = embs

      return entityLookup
