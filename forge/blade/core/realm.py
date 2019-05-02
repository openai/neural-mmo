import ray
import pickle
import numpy as np
from pdb import set_trace as T
import numpy as np

from forge import trinity as Trinity
from forge.trinity.spawn import Spawn
from forge.blade import entity, core
from itertools import chain
from copy import deepcopy

class Realm:
   def __init__(self, config, args, idx):
      #Random samples
      if config.SAMPLE:
         config = deepcopy(config)
         nent = np.random.randint(0, config.NENT)
         config.NENT = config.NPOP * (1 + nent // config.NPOP)
      self.world, self.desciples = core.Env(config, idx), {}
      self.config, self.args, self.tick = config, args, 0
      self.npop = config.NPOP

      self.env = self.world.env
      self.values = None

   def clientData(self):
      if self.values is None and hasattr(self, 'sword'):
         self.values = self.sword.anns[0].visVals()

      ret = {
            'environment': self.world.env,
            'entities': dict((k, v.packet()) for k, v in self.desciples.items()),
            'values': self.values
            }
      return pickle.dumps(ret)

   def spawn(self):
      if len(self.desciples) >= self.config.NENT:
         return

      entID, color = self.god.spawn()
      ent = entity.Player(entID, color, self.config)
      self.desciples[ent.entID] = ent

      r, c = ent.pos
      self.world.env.tiles[r, c].addEnt(entID, ent)
      self.world.env.tiles[r, c].counts[ent.population.val] += 1

   def cullDead(self, dead):
      for entID in dead:
         ent = self.desciples[entID]
         r, c = ent.pos
         self.world.env.tiles[r, c].delEnt(entID)
         self.god.cull(ent.annID)
         del self.desciples[entID]

   def stepWorld(self):
      ents = list(chain(self.desciples.values()))
      self.world.step(ents, [])

   def stepEnv(self):
      self.world.env.step()
      self.env = self.world.env.np()

   def getStim(self, ent):
      return self.world.env.stim(ent.pos, self.config.STIM)

class NativeRealm(Realm):
   def __init__(self, trinity, config, args, idx):
      super().__init__(config, args, idx)
      self.god = Spawn(config, args)
 
   def stepEnts(self):
      dead = []
      for ent in self.desciples.values():
         ent.step(self.world)

         if self.postmortem(ent, dead):
            continue

         stim = self.getStim(ent)
         actions = self.sword.decide(stim, ent)
         ent.act(self.world, actions)
         #self.stepEnt(ent, actions)

      self.cullDead(dead)

   def postmortem(self, ent, dead):
      entID = ent.entID
      if not ent.alive or ent.kill:
         dead.append(entID)
         if not self.config.TEST:
            self.sword.collectRollout(entID, ent)
         return True
      return False

   def step(self):
      self.spawn()
      self.stepEnv()
      self.stepEnts()
      self.stepWorld()

   def run(self, swordUpdate=None):
      self.recvSwordUpdate(swordUpdate)

      updates = None
      while updates is None:
         self.step()
         updates, logs = self.sword.sendUpdate()
      return updates, logs

class VecEnvRealm(Realm):
   #Use the default God behind the scenes for spawning
   def __init__(self, config, args, idx):
      super().__init__(config, args, idx)
      self.god = Spawn(config, args)

   def stepEnts(self, decisions):
      self.dead = []
      for tup in decisions:
         entID, actions = tup
         ent = self.desciples[entID]
         ent.step(self.world)

         if self.postmortem(ent, self.dead):
            continue

         ent.act(self.world, actions)

   def postmortem(self, ent, dead):
      entID = ent.entID
      if not ent.alive or ent.kill:
         dead.append(entID)
         return True
      return False

   def step(self, decisions):
      self.stepEnts(decisions)
      self.stepWorld()
      self.spawn()
      self.stepEnv()

      stims, rews, dones = [], [], []
      for entID, ent in self.desciples.items():
         stim = self.getStim(ent)
         stims.append((self.getStim(ent), ent))
         rews.append(1)

      self.cullDead(self.dead)
      return stims, rews, None, None

   def reset(self):
      return []


