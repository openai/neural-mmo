#Main world definition. Defines and manages entity handlers,
#Defines behavior of the world under all circumstances and handles
#interaction by agents. Also defines an accurate stimulus that
#encapsulates the world as seen by a particular agent

import numpy as np
from collections import defaultdict

from typing import Dict

from forge.blade import systems
from forge.blade.lib import utils
from forge.blade import lib

from forge.blade import core
from forge.blade.item import rawfish, knife, armor
from forge.blade.lib.enums import Palette
from forge.blade.entity.npc import NPC
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

   def spawn(self, entity) -> None:
      pos, entID = entity.pos, entity.entID
      self.realm.map.tiles[pos].addEnt(entity)
      self.entities[entID] = entity
 
   def cull(self) -> Dict[int, Player]:
      self.dead = {}
      for entID in list(self.entities):
         if not (player := self.entities[entID]).alive:
            r, c  = player.base.pos
            entID = player.entID
            self.dead[entID] = player

            self.realm.map.tiles[r, c].delEnt(entID)
            del self.entities[entID]
            self.realm.dataframe.remove(Static.Entity, entID, player.pos)

      return self.dead

   def preActions(self, decisions) -> None:
      '''Update agent history before performing actions
      Args:
         decisions: A dictionary of agent actions
      '''
      for entID, entity in self.entities.items():
         entity.history.update(self.realm, entity, decisions)

   def postActions(self, decisions) -> None:
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
 
   def spawn(self):
      for _ in range(self.config.NPC_SPAWN_ATTEMPTS):
         if len(self.entities) >= self.config.NMOB:
            break

         r, c = np.random.randint(0, self.config.TERRAIN_SIZE, 2).tolist()
         if not self.realm.map.tiles[r, c].habitable:
            continue

         if npc := NPC.spawn(self.realm, (r, c), self.idx):
            super().spawn(npc)
            self.idx -= 1

   def actions(self, realm):
      actions = {}
      for idx, entity in self.entities.items():
         actions[idx] = entity.decide(realm)
      return actions
       
class PlayerManager(EntityGroup):
   def __init__(self, realm, config, iden):
      super().__init__(config)
      self.palette = Palette(config.NPOP)
      self.iden    = iden

      self.realm   = realm
      self.idx     = 1

   def spawn(self):
      for _ in range(self.config.PLAYER_SPAWN_ATTEMPTS):
         if len(self.entities) >= self.config.NENT:
            break

         r, c   = self.config.SPAWN()
         if not self.realm.map.tiles[r, c].habitable:
            continue

         pop, name = self.iden()
         color     = self.palette.colors[pop]
         player    = Player(self.realm, (r, c), self.idx, pop, name, color)

         super().spawn(player)
         self.idx += 1

class Realm:
   def __init__(self, config, iden):
      #Load the world file
      self.dataframe = trinity.Dataframe(config)
      self.map       = core.Map(self, config)
      self.shape     = self.map.shape
      self.spawn     = config.SPAWN
      self.config    = config
      self.tick      = 0

      #Entity handlers
      self.stimSize = 3
      self.worldDim = 2*self.stimSize+1

      self.iden     = iden
      self.players  = PlayerManager(self, config, iden)
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

      self.envTimer  = utils.BenchmarkTimer()
      self.entTimer  = utils.BenchmarkTimer()
      self.cpuTimer  = utils.BenchmarkTimer()
      self.handlerTimer = utils.BenchmarkTimer()
      self.statTimer  = utils.BenchmarkTimer()

   def packet(self):
      return {'environment': self.map.np().tolist(),
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

   def reset(self, idx):
      self.map.reset(self, idx)
      self.tick = 0

      entities = {**self.players.entities, **self.npcs.entities}
      for entID, ent in entities.items():
         self.dataframe.remove(Static.Entity, entID, ent.pos)

      self.players  = PlayerManager(self, self.config, self.iden)
      self.npcs     = NPCManager(self, self.config)
  
   def step(self, decisions):
      self.players.spawn()
      while len(self.players.entities) == 0:
         self.players.spawn()

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
            ent = self.entity(entID)
            atn.call(self, ent, *args)

      self.players.postActions(decisions)
      self.npcs.postActions(npcDecisions)

      #Cull dead
      dead = self.players.cull()
      self.npcs.cull()

      #Update map
      self.map.step()

      self.tick += 1
      return dead


