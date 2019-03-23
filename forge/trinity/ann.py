from pdb import set_trace as T
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from forge.blade.action import tree
from forge.blade.action.tree import ActionTree
from forge.blade.action import action
from forge.blade.action.action import ActionRoot
from forge.blade.lib.enums import Neon
from forge.blade.lib import enums
from forge.ethyr import torch as torchlib
from forge.blade import entity

def classify(logits):
   if len(logits.shape) == 1:
      logits = logits.view(1, -1)
   distribution = Categorical(1e-3+F.softmax(logits, dim=1))
   atn = distribution.sample()
   return atn

class NetTree(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.modules = {}
      self.config = config
      self.h = config.HIDDEN

   def add(self, cls):
      if cls.argType is tree.ConstDiscrete:
         n = len(cls.edges)
         self.modules[cls] = EnvConstDiscrete(self.config, self.h, n)
      elif cls.argType is tree.VariableDiscrete:
         self.modules[cls] = EnvVariableDiscrete(self.config, self.h)

   def module(self, cls):
      if cls not in self.modules:
         cls = self.map(cls)
         self.add(cls)
      return self.modules[cls]

   #Force shared networks across particular nodes
   def map(self, cls):
      if cls in (action.Melee, action.Range, action.Mage):
         cls = action.Melee
      return cls

   def forward(self, env, ent, stim):
      actionTree = ActionTree(env, ent, ActionRoot)
      atn = actionTree.next()
      while atn is not None:
         module = self.module(atn)
         #Here's your issue: you funamentally have not handled arguments.
         #"args" is just "outs" -- you need to handle ACTUAL arguments for
         #variable discrete.
         assert atn.argType in (tree.ConstDiscrete, tree.VariableDiscrete)
         if atn.argType is tree.ConstDiscrete:
            nxtAtn, outs = module(env, ent, atn, stim)
            atn = actionTree.next(nxtAtn, outs=outs)
         elif atn.argType is tree.VariableDiscrete:
            argument, outs = module(env, ent, atn, stim)
            T()
            atn = actionTree.next(atn, argument, outs)
      
      atns, outs = actionTree.unpackActions()
      T()
      return atns, outs

class EnvConstDiscrete(nn.Module):
   def __init__(self, config, h, ydim):
      super().__init__()
      self.envNet = Env(config)
      self.constDiscrete = ConstDiscrete(config, h, ydim)

   def forward(self, env, ent, action, stim):
      stim = self.envNet(stim.conv, stim.flat, stim.ents) 
      action, atn, atnIdx = self.constDiscrete(env, ent, action, stim)
      return action, (atn, atnIdx)

class EnvVariableDiscrete(nn.Module):
   def __init__(self, config, h):
      super().__init__()
      self.envNet = Env(config)
      self.variableDiscrete = VariableDiscrete(config, 2*h, h)
      self.config = config

      entDim = 11
      self.targEmbed  = Ent(entDim, h)

   def forward(self, env, ent, action, stim):
      stim = self.envNet(stim.conv, stim.flat, stim.ents) 
      #action, atn, atnIdx = self.variableDiscrete(env, ent, action, stim)

      #Embed targets
      targets = action.args(env, ent, self.config)
      targets = torch.tensor([e.stim for e in targets]).float()
      targets = self.targEmbed(targets).unsqueeze(0)

      argument, arg, argIdx = self.variableDiscrete(
            env, ent, action, stim, targets)
      return argument, (arg, argIdx)
 
####### Network Modules
class ConstDiscrete(nn.Module):
   def __init__(self, config, h, ydim):
      super().__init__()
      self.fc1 = torch.nn.Linear(h, ydim)
      self.config = config

   def forward(self, env, ent, action, stim):
      leaves = action.args(env, ent, self.config)
      x = self.fc1(stim)
      xIdx = classify(x)
      try:
         leaf = leaves[int(xIdx)]
      except:
         T()
      return leaf, x, xIdx

class VariableDiscrete(nn.Module):
   def __init__(self, config, xdim, h):
      super().__init__()
      self.attn = AttnCat(xdim, h)
      self.config = config

   #Arguments: stim, action/argument embedding
   def forward(self, env, ent, action, key, vals):
      leaves = action.args(env, ent, self.config)
      x = self.attn(key, vals)
      xIdx = classify(x)
      leaf = leaves[int(xIdx)]
      return leaf, x, xIdx 

class AttnCat(nn.Module):
   def __init__(self, xdim, h):
      super().__init__()
      #self.fc1 = torch.nn.Linear(xdim, h)
      #self.fc2 = torch.nn.Linear(h, 1)
      self.fc = torch.nn.Linear(xdim, 1)
      self.h = h

   def forward(self, x, args):
      n = args.shape[0]
      x = x.expand(n, self.h)
      xargs = torch.cat((x, args), dim=1)

      x = self.fc(xargs)
      #x = F.relu(self.fc1(xargs))
      #x = self.fc2(x)
      return x.view(1, -1)
####### End network modules

class ValNet(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.fc = torch.nn.Linear(config.HIDDEN, 1)
      self.envNet = Env(config)

   def forward(self, conv, flat, ent):
      stim = self.envNet(conv, flat, ent) 
      x = self.fc(stim)
      x = x.view(1, -1)
      return x

class Ent(nn.Module):
   def __init__(self, entDim, h):
      super().__init__()
      self.ent = torch.nn.Linear(entDim, h)
 
   def forward(self, ents):
      ents = self.ent(ents)
      ents, _ = torch.max(ents, 0)
      return ents

class Env(nn.Module):
   def __init__(self, config):
      super().__init__()
      h = config.HIDDEN
      entDim = 11 # + 225

      self.fc1  = torch.nn.Linear(3*h, h)
      self.embed = torch.nn.Embedding(7, 7)

      self.conv = torch.nn.Linear(1800, h)
      self.flat = torch.nn.Linear(entDim, h)
      self.ents = Ent(entDim, h)

   def forward(self, conv, flat, ents):
      tiles, nents = conv[0], conv[1]
      nents = nents.view(-1)

      tiles = self.embed(tiles.view(-1).long()).view(-1)
      conv = torch.cat((tiles, nents))

      conv = self.conv(conv)
      ents = self.ents(ents)
      flat = self.flat(flat)

      x = torch.cat((conv, flat, ents)).view(1, -1)
      x = self.fc1(x)
      #Removed relu (easier training, lower policy cap)
      #x = torch.nn.functional.relu(self.fc1(x))
      return x

class MoveNet(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.moveNet = ConstDiscrete(config, config.HIDDEN, 5)
      self.envNet = Env(config)

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
      self.targEmbed  = Ent(entDim, h)
      self.h = h

      self.envNet = Env(config)
      self.styleNet = ConstDiscrete(config, h, 3)
      self.targNet  = VariableDiscrete(config, 3*h, h)

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
      self.valNet = ValNet(config)

      self.config = config
      self.actionNet  = NetTree(config)
      self.moveNet    = MoveNet(config)
      self.attackNet  = (StyleAttackNet(config) if 
            config.AUTO_TARGET else AttackNet(config))

   def forward(self, ent, env):
      s = torchlib.Stim(ent, env, self.config)
      val  = self.valNet(s.conv, s.flat, s.ents)

      atns, outs = self.actionNet(env, ent, s)
      T()
      move   = actions[action.Move]
      attack = actions[action.Attack]
      target = actions[action.Melee]

      moveArg, moveOuts = move
      attkArg, attkOuts = attack
      targArg, targOuts = target

      actions = (action.Move, attkArg)
      arguments = (moveArg, [targArg])
      outs = (moveOuts, attkOuts, targOuts)
      
      '''
      #Actions
      actions = ActionTree(env, ent, ActionRoot).actions()
      move, attk = actions

      moveArg, moveOuts = self.moveNet(
            env, ent, move, s)
      attk, attkArg, attkOuts = self.attackNet(
            env, ent, attk, s)
      actions    = (move, attk)
      arguments = (moveArg, attkArg)
      outs      = (moveOuts, *attkOuts)
      '''

      return actions, arguments, outs, val

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
