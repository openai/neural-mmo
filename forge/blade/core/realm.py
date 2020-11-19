#Main world definition. Defines and manages entity handlers,
#Defines behavior of the world under all circumstances and handles
#interaction by agents. Also defines an accurate stimulus that
#encapsulates the world as seen by a particular agent

import numpy as np
from collections import defaultdict

from forge.blade import systems
from forge.blade.lib import utils
from forge.blade import lib

from forge.blade import core
from forge.blade.item import rawfish, knife, armor
from forge.blade.lib.enums import Palette
from forge.blade.entity import npc
from pdb import set_trace as T
from forge.blade.entity import Player
from forge import trinity

from forge.blade.io.stimulus import Static

#actions[atn.priority][entID] = [atn, args]
def prioritized(entities, merged):
   for idx, actions in entities.items():
      for atn, args in actions.items():
         merged[atn.priority].append((idx, (atn, args.values())))
   return merged

class EntityGroup:
   def __init__(self, config):
      self.config   = config
      self.entities = {}
      self.dead     = {}

   def packet(self):
      alive = dict((k, v.packet()) for k, v in self.entities.items())
      dead  = dict((k, v.packet()) for k, v in self.dead.items())
      return {**alive, **dead}

   def items(self):
      return self.entities.items()

   def __getitem__(self, key):
      return self.entities[key]

   def add(iden, entity):
      assert iden not in self.entities
      self.entities[iden] = entity

   def remove(iden):
      assert iden in self.entities
      del self.entities[iden] 

   def __contains__(self, e):
      return e in self.entities

   def __len__(self):
      return len(self.entities)

   def cull(self):
      #Cull dead players
      dead = {}
      for entID in list(self.entities):
         player = self.entities[entID]
         if not player.alive:
            r, c  = player.base.pos
            entID = player.entID
            dead[entID] = player

            self.realm.map.tiles[r, c].delEnt(entID)
            del self.entities[entID]
            self.realm.dataframe.remove(Static.Entity, entID, player.pos)

      self.dead = dead
      return dead

   def preActions(self, decisions):
      '''Update agent history before performing actions
      Args:
         decisions: A dictionary of agent actions
      '''
      for entID, entity in self.entities.items():
         entity.history.update(self.realm, entity, decisions)

   def postActions(self, decisions):
      '''Update agent data after performing actions
      Args:
         decisions: A dictionary of agent actions
      '''
      for entID, entity in self.entities.items():
         entity.update(self.realm, decisions)

class NPCManager(EntityGroup):
   def __init__(self, realm, config):
      super().__init__(config)
      self.realm = realm
      self.idx   = -1
 
   def spawn(self, nTries=25):
      R, C = self.realm.shape
      for i in range(nTries):
         if len(self.entities) >= self.config.NMOB:
            break

         r = np.random.randint(0, R)
         c = np.random.randint(0, C)

         if len(self.realm.map.tiles[r, c].ents) != 0:
            continue

         tile = self.realm.map.tiles[r, c]
         if tile.mat.tex != 'grass':
            continue

         entity = npc.NPC.spawn(self.realm, (r, c), self.idx)
         self.entities[self.idx] = entity
         self.realm.map.tiles[r, c].addEnt(self.idx, entity)
         self.idx -= 1

   def actions(self, realm):
      actions = {}
      for idx, entity in self.entities.items():
         actions[idx] = entity.decide(realm)
      return actions
       
class PlayerManager(EntityGroup):
   def __init__(self, realm, config):
      super().__init__(config)
      self.realm = realm

      self.palette = Palette(config.NPOP)

   def spawn(self, iden, pop, name):
      assert len(self.entities) <= self.config.NENT
      assert iden is not None
 
      if len(self.entities) == self.config.NENT:
         return

      r, c   = self.config.SPAWN()
      while len(self.realm.map.tiles[r, c].ents) != 0:
         r, c   = self.config.SPAWN()

      color  = self.palette.colors[pop]
      player = Player(self.realm, (r, c), iden, pop, name, color)

      self.realm.map.tiles[r, c].addEnt(iden, player)
      self.entities[player.entID] = player

class Realm:
   def __init__(self, config, idx):
      #Load the world file
      self.dataframe = trinity.Dataframe(config)
      self.map       = core.Map(self, config, idx)
      self.shape     = self.map.shape
      self.spawn     = config.SPAWN
      self.config    = config

      #Entity handlers
      self.stimSize = 3
      self.worldDim = 2*self.stimSize+1
      self.players  = PlayerManager(self, config)
      self.npcs     = NPCManager(self, config)

      #Exchange - For future updates
      self.market = systems.Exchange()
      sardine = rawfish.Sardine
      nife = knife.Iron
      armr = armor.Iron
      self.market.buy(sardine, 10, 30)
      self.market.sell(nife, 20, 50)
      self.market.buy(armr, 1, 500)
      self.market.sell(armr, 3, 700)

      self.stats = lib.StatTraker()

      self.tick = 0
      self.envTimer  = utils.BenchmarkTimer()
      self.entTimer  = utils.BenchmarkTimer()
      self.cpuTimer  = utils.BenchmarkTimer()
      self.handlerTimer = utils.BenchmarkTimer()
      self.statTimer  = utils.BenchmarkTimer()

   def packet(self):
      return {'environment': self.map,
              'resource': self.map.packet(),
              'player': self.players.packet(),
              'npc': self.npcs.packet()}

   @property
   def nEntities(self):
      return len(self.players.entities)

   def entity(self, entID):
      if entID < 0:
         return self.npcs[entID]
      else:
         return self.players[entID]

   #Hook for render
   def graphicsData(self):
      return self.env, self.stats

   def step(self, decisions):
      #NPC Spawning and decisions
      self.npcs.spawn()
      npcDecisions = self.npcs.actions(self)

      #Prioritize actions
      merged       = defaultdict(list)
      prioritized(decisions, merged)
      prioritized(npcDecisions, merged)

      #Update entities and perform actions
      self.players.preActions(decisions)
      self.npcs.preActions(npcDecisions)

      for priority in sorted(merged):
         for entID, (atn, args) in merged[priority]:
            atn.call(self, self.entity(entID), *args)

      self.players.postActions(decisions)
      self.npcs.postActions(npcDecisions)

      #Cull dead
      dead = self.players.cull()
      self.npcs.cull()

      #Update map
      self.map.step()

      self.tick += 1
      return dead


