from pdb import set_trace as T
import numpy as np

from collections import defaultdict, Mapping
from typing import Dict, Callable

from neural_mmo.forge.blade import core
from neural_mmo.forge.blade.lib.enums import Palette
from neural_mmo.forge.blade.entity.npc import NPC
from neural_mmo.forge.blade.entity import Player
from neural_mmo.forge import trinity

from neural_mmo.forge.blade.io.stimulus import Static

def prioritized(entities: Dict, merged: Dict):
   '''Sort actions into merged according to priority'''
   for idx, actions in entities.items():
      for atn, args in actions.items():
         merged[atn.priority].append((idx, (atn, args.values())))
   return merged

class EntityGroup(Mapping):
   def __init__(self, config, realm):
      self.dataframe = realm.dataframe
      self.config    = config

      self.entities  = {}
      self.dead      = {}

   def __len__(self):
      return len(self.entities)

   def __contains__(self, e):
      return e in self.entities

   def __getitem__(self, key):
      return self.entities[key]
   
   def __iter__(self):
      yield from self.entities

   def items(self):
      return self.entities.items()

   @property
   def corporeal(self):
      return {**self.entities, **self.dead}

   @property
   def packet(self):
      return {k: v.packet() for k, v in self.corporeal.items()}

   def reset(self):
      for entID, ent in self.entities.items():
         self.dataframe.remove(Static.Entity, entID, ent.pos)

      self.spawned  = False
      self.entities = {}
      self.dead     = {}

   def add(iden, entity):
      assert iden not in self.entities
      self.entities[iden] = entity

   def remove(iden):
      assert iden in self.entities
      del self.entities[iden] 

   def spawn(self, entity):
      pos, entID = entity.pos, entity.entID
      self.realm.map.tiles[pos].addEnt(entity)
      self.entities[entID] = entity
 
   def cull(self):
      self.dead = {}
      for entID in list(self.entities):
         player = self.entities[entID]
         if not player.alive:
            r, c  = player.base.pos
            entID = player.entID
            self.dead[entID] = player

            self.realm.map.tiles[r, c].delEnt(entID)
            del self.entities[entID]
            self.realm.dataframe.remove(Static.Entity, entID, player.pos)

      return self.dead

   def update(self, actions):
      for entID, entity in self.entities.items():
         entity.update(self.realm, actions)

class NPCManager(EntityGroup):
   def __init__(self, config, realm):
      super().__init__(config, realm)
      self.realm = realm
      self.idx   = -1
 
   def spawn(self):
      if not self.config.game_system_enabled('NPC'):
         return

      for _ in range(self.config.NPC_SPAWN_ATTEMPTS):
         if len(self.entities) >= self.config.NMOB:
            break

         center = self.config.TERRAIN_CENTER
         border = self.config.TERRAIN_BORDER
         r, c   = np.random.randint(border, center+border, 2).tolist()
         if self.realm.map.tiles[r, c].occupied:
            continue

         npc = NPC.spawn(self.realm, (r, c), self.idx)
         if npc: 
            super().spawn(npc)
            self.idx -= 1

   def actions(self, realm):
      actions = {}
      for idx, entity in self.entities.items():
         actions[idx] = entity.decide(realm)
      return actions
       
class PlayerManager(EntityGroup):
   def __init__(self, config, realm):
      super().__init__(config, realm)

      self.palette = Palette(config.NPOP)
      self.agents = config.AGENT_LOADER(config.AGENTS) 
      self.realm  = realm
      self.idx    = 1

   def spawnIndividual(self, r, c):
      agent  = next(self.agents)(self.config, self.idx)
      player = Player(self.realm, (r, c), agent)
      super().spawn(player)
      self.idx += 1

   def spawn(self):
      if self.config.SPAWN == self.config.SPAWN_CONCURRENT:
         if self.spawned:
            return 

         self.spawned = True
         for r, c in self.config.SPAWN():
            assert not self.realm.map.tiles[r, c].occupied
            self.spawnIndividual(r, c)
         return
          
      #MMO-style spawning
      for _ in range(self.config.PLAYER_SPAWN_ATTEMPTS):
         if len(self.entities) >= self.config.NENT:
            break

         r, c   = self.config.SPAWN()
         if self.realm.map.tiles[r, c].occupied:
            continue

         self.spawnIndividual(r, c)

      while len(self.entities) == 0:
         self.spawn()

class Realm:
   '''Top-level world object'''
   def __init__(self, config):
      self.config   = config

      #Load the world file
      self.dataframe = trinity.Dataframe(config)
      self.map       = core.Map(config, self)

      #Entity handlers
      self.players  = PlayerManager(config, self)
      self.npcs     = NPCManager(config, self)

   def reset(self, idx):
      '''Reset the environment and load the specified map

      Args:
         idx: Map index to load
      ''' 
      self.map.reset(self, idx)
      self.players.reset()
      self.npcs.reset()
      self.tick = 0
 
   def packet(self):
      '''Client packet'''
      return {'environment': self.map.repr,
              'resource': self.map.packet,
              'player': self.players.packet,
              'npc': self.npcs.packet}

   @property
   def population(self):
      '''Number of player agents'''
      return len(self.players.entities)

   def entity(self, entID):
      '''Get entity by ID'''
      if entID < 0:
         return self.npcs[entID]
      else:
         return self.players[entID]

   def step(self, actions):
      '''Run game logic for one tick
      
      Args:
         actions: Dict of agent actions
      '''
      #Prioritize actions
      npcActions = self.npcs.actions(self)
      merged     = defaultdict(list)
      prioritized(actions, merged)
      prioritized(npcActions, merged)

      #Update entities and perform actions
      self.players.update(actions)
      self.npcs.update(npcActions)

      #Execute actions
      for priority in sorted(merged):
         for entID, (atn, args) in merged[priority]:
            ent = self.entity(entID)
            atn.call(self, ent, *args)

      #Cull dead agents and spawn new ones
      dead = self.players.cull()
      self.npcs.cull()

      self.players.spawn()
      self.npcs.spawn()

      #Update map
      self.map.step()
      self.tick += 1

      return dead
