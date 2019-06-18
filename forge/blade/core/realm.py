import ray
import pickle
import numpy as np
from pdb import set_trace as T
import numpy as np

from forge.blade import entity, core

from collections import defaultdict
from itertools import chain
from copy import deepcopy

from forge.blade.lib.enums import Palette
from forge.trinity.timed import runtime, Timed
from forge.blade.io.action import Static

class Spawner:
   def __init__(self, config, args):
      self.config, self.args = config, args

      self.nEnt, self.nPop = config.NENT, config.NPOP
      self.popSz = self.nEnt // self.nPop

      self.ents = 0
      self.pops = defaultdict(int)
      self.palette = Palette(self.nPop)

   def spawn(self, realm, iden, pop, name):
      assert self.pops[pop] <= self.popSz
      assert self.ents      <= self.nEnt

      #Too many total entities
      if self.ents == self.nEnt:
         return None

      if self.pops[pop] == self.popSz:
         return None

      self.pops[pop] += 1
      self.ents += 1

      color = self.palette.colors[pop]
      ent   = entity.Player(self.config, iden, pop, name, color)
      assert ent not in realm.desciples

      r, c = ent.pos
      realm.desciples[ent.entID] = ent
      realm.world.env.tiles[r, c].addEnt(iden, ent)
      realm.world.env.tiles[r, c].counts[ent.population.val] += 1

   def cull(self, pop):
      assert self.pops[pop] >= 1
      assert self.ents      >= 1

      self.pops[pop] -= 1
      self.ents -= 1

      if self.pops[pop] == 0:
         del self.pops[pop]

class Realm(Timed):
   '''Neural MMO Environment'''

  
   def __init__(self, config, args, idx):
      '''Initializer
      
      Args:
         config: A Configuration object
         args: Hook for command line arguments
         idx: Index of the world file to load

      '''
 
      super().__init__()
      #Random samples
      if config.SAMPLE:
         config = deepcopy(config)
         nent = np.random.randint(0, config.NENT)
         config.NENT = config.NPOP * (1 + nent // config.NPOP)

      self.spawner = Spawner(config, args)
      self.world, self.desciples = core.Env(config, idx), {}
      self.config, self.args = config, args
      self.npop = config.NPOP

      self.worldIdx = idx
      self.tick = 0

      self.env = self.world.env
      self.values = None

   def clientData(self):
      '''Data packet used by the renderer'''
 
      if self.values is None:# and hasattr(self, 'sword'):
         #self.values = self.sword.anns[0].visVals()
         self.values = []

      ret = {
            'environment': self.world.env,
            'entities': dict((k, v.packet()) for k, v in self.desciples.items()),
            'values': self.values
            }
      return pickle.dumps(ret)

   def cullDead(self, dead):
      for entID in dead:
         ent = self.desciples[entID]
         r, c = ent.pos
         self.world.env.tiles[r, c].delEnt(entID)
         self.spawner.cull(ent.annID)
         del self.desciples[entID]

   def stepWorld(self):
      ents = list(chain(self.desciples.values()))
      self.world.step(ents, [])

   def stepEnv(self):
      self.world.env.step()
      self.env = self.world.env.np()

   def getStim(self, ent):
      return self.world.env.stim(ent.pos, self.config.STIM)

   #Only saves the first action of each priority
   def prioritize(self, decisions):
      actions = defaultdict(dict)
      for tup in decisions:
         entID, atns = tup
         for atnArgs in reversed(atns):
            priority = atnArgs.action.priority
            actions[priority][entID] = atnArgs
      return actions

   #Take actions
   def act(self, actions):
      for priority, tups in actions.items():
         for entID, atnArgs in tups.items():
            ent = self.desciples[entID]
            ent.act(self.world, atnArgs)

   def stepEnts(self, decisions):
      #Step all ents first
      for tup in decisions:
         entID, actions = tup
         ent = self.desciples[entID]
         ent.step(self.world, actions)

      actions = self.prioritize(decisions)
      self.act(actions)

      #Finally cull dead. This will enable MAD melee
      rewards, dead = [], []
      for tup in decisions:
         entID, actions = tup
         ent = self.desciples[entID]

         if self.postmortem(ent, dead):
            rewards.append(-1)
            continue

         rewards.append(0)

      self.cullDead(dead)
      return rewards

   def postmortem(self, ent, dead):
      entID = ent.entID
      if not ent.alive or ent.kill:
         dead.append(entID)
         return True
      return False

   def getStims(self, rewards):
      stims = []
      for entID, ent in self.desciples.items():
         tile = self.world.env.tiles[ent.r.val, ent.c.val].tex
         stim = self.getStim(ent)
         stims.append((self.getStim(ent), ent))

      return stims, rewards, None, None

   @runtime
   def step(self, decisions):
      '''Take actions for all agents and return new observations

      Args:
         List of decisions

      Returns:
         observations, rewards, None, None
      '''

      self.tick += 1

      rewards = self.stepEnts(decisions)
      self.stepWorld()

      iden, pop, name = self.spawn()
      self.spawner.spawn(self, iden, pop, name)

      self.stepEnv()
      return self.getStims(rewards)

   def reset(self):
      '''Stub for conformity with Gym that returns an empty list'''

      return []


