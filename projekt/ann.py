'''Demo agent class

The policies for v1.2 are sanity checyks only.
I rushed a simple baseline to get the new client
out faster. I am working on something better now.
If you want help getting other models working in
the meanwhile, drop in the Discord support channel.'''
from pdb import set_trace as T
import numpy as np
import torch

from torch import nn

from forge import trinity

from forge.blade.lib.enums import Neon
from forge.blade.lib import enums
from forge.ethyr import torch as torchlib

from forge.ethyr.torch import policy
from forge.blade import entity

from forge.ethyr.torch.param import setParameters, getParameters, zeroGrads
from forge.ethyr.torch import param

from forge.ethyr.torch.policy import functional
from forge.ethyr.io.utils import pack, unpack

from forge.ethyr.torch.io.stimulus import Env
from forge.ethyr.torch.io.action import NetTree
from forge.ethyr.torch.policy import attention

class Atn(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.fc = nn.Linear(config.HIDDEN, 4)

   def forward(self, x):
      x    = self.fc(x)
      xIdx = functional.classify(x)

      x = [[e] for e in x]
      xIdx = xIdx.view(-1, 1)

      return x, xIdx
 
#Hacky inner attention layer
class EmbAttn(nn.Module):
   def __init__(self, config, n):
      super().__init__()
      self.fc1 = nn.Linear(n*config.EMBED, config.HIDDEN)

   def forward(self, x):
      batch, ents, _, _ = x.shape
      x = x.view(batch, ents, -1)
      x = self.fc1(x)
      return x

#Hacky outer attention layer
class EntAttn(nn.Module):
   def __init__(self, xDim, yDim, n):
      super().__init__()
      self.fc1 = nn.Linear(n*xDim, yDim)
      #self.emb = attention.MaxReluBlock(128)

   def forward(self, x):
      batch, ents, _, = x.shape
      x = x.view(batch, -1)
      x = self.fc1(x)
      #x = self.emb(x)
      return x

#Hacky outer attention layer
class TileConv(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.h = h

      self.conv1 = nn.Conv2d(h, h, 3)
      self.pool1 = nn.MaxPool2d(2)
      self.fc1 = nn.Linear(h*6*6, h)

   def forward(self, x):
      x = x.transpose(1, 2)
      x = x.view(-1, self.h, 15, 15)
      x = self.conv1(x)
      x = self.pool1(x)

      batch, _, _, _ = x.shape
      x = x.view(batch, -1)
      x = self.fc1(x)

      return x


#Variable number of entities
class Entity(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.emb = attention.Attention(config.EMBED, config.HIDDEN)
      self.ent = attention.Attention(config.HIDDEN, config.HIDDEN)

#Fixed number of entities
class Tile(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.emb = attention.Attention(config.EMBED, config.HIDDEN)
      #self.ent = EntAttn(config.HIDDEN, config.HIDDEN, 225)
      self.ent = TileConv(config.HIDDEN)

class Net(nn.Module):
   def __init__(self, config):
      super().__init__()
      h = config.HIDDEN

      self.attns = nn.ModuleDict({
         'Tile':   Tile(config),
         'Entity': Entity(config),
         'Meta':   EntAttn(h, h, 2),
      })

      self.val  = torch.nn.Linear(h, 1)

class ANN(nn.Module):
   def __init__(self, config):
      '''Demo model'''
      super().__init__()
      self.config = config
      self.net = nn.ModuleList([Net(config)
            for _ in range(config.NPOP)])

      #Shared environment/action maps
      self.env    = Env(config)
      self.action = NetTree(config)
      #self.atn = Atn(config)

   def forward(self, pop, stim, actions):
      net           = self.net[pop]
      stim, embed   = self.env(net, stim)
      val           = net.val(stim)

      atns, atnsIdx = self.action(stim, actions, embed)
      #atns, atnsIdx = self.atn(stim)

      return atns, atnsIdx, val

   def recvUpdate(self, update):
      setParameters(self, update)

   def grads(self):
      grads = param.getGrads(self)
      zeroGrads(self)
      return grads

   def params(self):
      return param.getParameters(self)

   #These hooks are outdated. Better policy
   #visualization for the new client is planned
   #for a future update
   def visDeps(self):
      from forge.blade.core import realm
      from forge.blade.core.tile import Tile
      colorInd = int(self.config.NPOP*np.random.rand())
      color    = Neon.color12()[colorInd]
      color    = (colorInd, color)
      ent = realm.Desciple(-1, self.config, color).server
      targ = realm.Desciple(-1, self.config, color).server

      sz = 15
      tiles = np.zeros((sz, sz), dtype=object)
      for r in range(sz):
         for c in range(sz):
            tiles[r, c] = Tile(enums.Grass, r, c, 1, None)

      targ.pos = (7, 7)
      tiles[7, 7].addEnt(0, targ)
      posList, vals = [], []
      for r in range(sz):
         for c in range(sz):
            ent.pos  = (r, c)
            tiles[r, c].addEnt(1, ent)
            #_, _, val = self.net(tiles, ent)
            val = np.random.rand()
            vals.append(float(val))
            tiles[r, c].delEnt(1)
            posList.append((r, c))
      vals = list(zip(posList, vals))
      return vals

   #These hooks are outdated. Better policy
   #visualization for the new client is planned
   #for a future update
   def visVals(self, food='max', water='max'):
      from forge.blade.core import realm
      posList, vals = [], []
      R, C = self.world.shape
      for r in range(self.config.BORDER, R-self.config.BORDER):
          for c in range(self.config.BORDER, C-self.config.BORDER):
            colorInd = int(self.config.NPOP*np.random.rand())
            color    = Neon.color12()[colorInd]
            color    = (colorInd, color)
            ent = entity.Player(-1, color, self.config)
            ent._r.update(r)
            ent._c.update(c)
            if food != 'max':
               ent._food = food
            if water != 'max':
               ent._water = water
            posList.append(ent.pos)

            self.world.env.tiles[r, c].addEnt(ent.entID, ent)
            stim = self.world.env.stim(ent.pos, self.config.STIM)
            #_, _, val = self.net(stim, ent)
            val = np.random.rand()
            self.world.env.tiles[r, c].delEnt(ent.entID)
            vals.append(float(val))

      vals = list(zip(posList, vals))
      return vals
