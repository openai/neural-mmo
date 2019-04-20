from pdb import set_trace as T
import numpy as np
import torch

from torch import nn

from forge.blade.action import tree
from forge.blade.action.tree import ActionTree
from forge.blade.action import action
from forge.blade.action.action import ActionRoot, NodeType
from forge.blade.lib.enums import Neon
from forge.blade.lib import enums
from forge.ethyr import torch as torchlib
from forge.ethyr.torch import policy, newpolicy
from forge.blade import entity

from forge.ethyr.torch.netgen.stim import Env

class MoveNet(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.moveNet = policy.ConstDiscrete(config, config.HIDDEN, 5)
      self.envNet = policy.Env(config)

   def forward(self, env, ent, action, s):
      stim = self.envNet(s.conv, s.flat, s.ents)
      action, arg, argIdx = self.moveNet(env, ent, action, stim)
      return action, (arg, argIdx)

#Network that selects an attack style
class StyleAttackNet(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.config, h = config, config.HIDDEN
      self.h = h

      self.envNet = Env(config)
      self.targNet = ConstDiscrete(config, h, 3)

   def target(self, ent, arguments):
      if len(arguments) == 1:
          return arguments[0]
      arguments = [e for e in arguments if e.entID != ent.entID]
      arguments = sorted(arguments, key=lambda a: a.health.val)
      return arguments[0]

   def forward(self, env, ent, action, s):
      stim = self.envNet(s.conv, s.flat, s.ents)
      action, atn, atnIdx = self.targNet(env, ent, action, stim)

      #Hardcoded targeting
      arguments = action.args(env, ent, self.config)
      argument = self.target(ent, arguments)

      attkOuts = [(atn, atnIdx)]
      return action, [argument], attkOuts

#Network that selects an attack and target (In progress,
#for learned targeting experiments)
class AttackNet(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.config, h = config, config.HIDDEN
      entDim = 11
      self.styleEmbed = torch.nn.Embedding(3, h)
      self.targEmbed  = policy.Ent(entDim, h)
      self.h = h

      self.envNet = policy.Env(config)
      self.styleNet = policy.ConstDiscrete(config, h, 3)
      self.targNet  = policy.VariableDiscrete(config, 3*h, h)

   def forward(self, env, ent, action, s):
      stim = self.envNet(s.conv, s.flat, s.ents)
      action, atn, atnIdx = self.styleNet(env, ent, action, stim)

      #Embed targets
      targets = action.args(env, ent, self.config)
      targets = torch.tensor([e.stim for e in targets]).float()
      targets = self.targEmbed(targets).unsqueeze(0)
      nTargs  = len(targets)

      atns    = self.styleEmbed(atnIdx).expand(nTargs, self.h)
      vals    = torch.cat((atns, targets), 1)

      argument, arg, argIdx = self.targNet(
            env, ent, action, stim, vals)

      attkOuts = ((atn, atnIdx), (arg, argIdx))
      return action, [argument], attkOuts

class ANN(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.net = newpolicy.Net(config)
      self.config = config

   def forward(self, env, ent):
      atns, outs, val = self.net(env, ent)
      return atns, outs, val

   #Messy hooks for visualizers
   def visDeps(self):
      from forge.blade.core import realm
      from forge.blade.core.tile import Tile
      colorInd = int(12*np.random.rand())
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
            s = torchlib.Stim(ent, tiles, self.config)
            conv, flat, ents = s.conv, s.flat, s.ents
            val  = self.valNet(conv, s.flat, s.ents)
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
            colorInd = int(12*np.random.rand())
            color    = Neon.color12()[colorInd]
            color    = (colorInd, color)
            ent = entity.Player(-1, color, self.config)
            ent._pos = (r, c)

            if food != 'max':
               ent._food = food
            if water != 'max':
               ent._water = water
            posList.append(ent.pos)

            self.world.env.tiles[r, c].addEnt(ent.entID, ent)
            stim = self.world.env.stim(ent.pos, self.config.STIM)
            s = torchlib.Stim(ent, stim, self.config)
            val = self.valNet(s.conv, s.flat, s.ents).detach()
            self.world.env.tiles[r, c].delEnt(ent.entID)
            vals.append(float(val))

      vals = list(zip(posList, vals))
      return vals
