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

from forge.ethyr.torch.netgen.stim import Env
from forge.ethyr.torch.param import setParameters, getParameters, zeroGrads
from forge.ethyr.torch import param

from forge.ethyr.torch.policy import Net 
from forge.ethyr.torch.netgen.stim import Env
from forge.ethyr.torch.netgen.action import NetTree

class ANN(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.config = config
      self.net = nn.ModuleList([Net(config)
            for _ in range(config.NPOP)])
      self.env    = Env(config)
      self.action = NetTree(config)

   #TODO: Need to select net index
   def forward(self, stim, obs=None, actions=None):
      #Add in action processing to input? Or maybe output embed?
      stim, embed = self.env(self.net[0].net, stim)
      val         = self.net[0].val(stim)

      atns, outs = self.action(stim, embed, obs, actions)
      return atns, outs, val

   def recvUpdate(self, update):
      setParameters(self.net, update)
      zeroGrads(self.net)

   def grads(self):
      return param.getGrads(self)

   def params(self):
      return param.getParameters(self)

   #Messy hooks for visualizers
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
