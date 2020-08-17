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

class NPCManager(EntityGroup):
   def __init__(self, realm, config):
      super().__init__(config)
      self.realm = realm
      self.idx   = -1
 
   def spawn(self, nTries=25):
      R, C = self.realm.shape
      for i in range(nTries):
         if len(self.entities) > self.config.NMOB:
            break

         r = np.random.randint(0, R)
         c = np.random.randint(0, C)
         tile = self.realm.map.tiles[r, c]

         if tile.mat.tex != 'grass':
            continue

         entity = npc.NPC.spawn(self.config, (r, c), self.idx)
         self.entities[self.idx] = entity
         self.realm.map.tiles[r, c].addEnt(self.idx, entity)
         self.idx -= 1

   def actions(self, realm):
      actions = {}
      for idx, entity in self.entities.items():
         actions[idx] = entity.step(realm)
      return actions
       
   def cull(self):
      #Cull dead players
      dead = set()
      for entID in list(self.entities):
         player = self.entities[entID]
         if not player.alive:
            r, c  = player.base.pos
            entID = player.entID
            dead.add(player)

            self.realm.map.tiles[r, c].delEnt(entID)
            del self.entities[entID]

      return dead


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
      color  = self.palette.colors[pop]
      player = Player(self.config, (r, c), iden, pop, name, color)

      self.realm.map.tiles[r, c].addEnt(iden, player)
      self.entities[player.entID] = player

   def step(self, decisions):
      '''Perform agent actions
      Args:
         decisions: A dictionary of agent actions

      Returns:
         dead: A list of dead agents
      '''
      #Update players before performing actions
      for entID, actions in decisions.items():
         ent = self.entities[entID]
         ent.step(self.realm, actions)   

   def cull(self):
      #Cull dead players
      dead = set()
      for entID in list(self.entities):
         player = self.entities[entID]
         if not player.alive:
            r, c  = player.base.pos
            entID = player.entID
            dead.add(player)

            self.realm.map.tiles[r, c].delEnt(entID)
            del self.entities[entID]

      return dead

class Realm:
   def __init__(self, config, idx):
      #Load the world file
      self.map    = core.Map(config, idx)
      self.shape  = self.map.shape
      self.spawn  = config.SPAWN
      self.config = config

      #Entity handlers
      self.stimSize = 3
      self.worldDim = 2*self.stimSize+1
      self.players = PlayerManager(self, config)
      self.npcs    = NPCManager(self, config)

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

      self.mobIdx = 0
      self.mobs = {}

   def stim(self, pos):
      return self.env.getPadded(self.env.tiles, pos, 
            self.stimSize, key=lambda e:e.index).astype(np.int8)

   #Hook for render
   def graphicsData(self):
      return self.env, self.stats

   def step(self, decisions):
      self.npcs.spawn()

      #Prioritize actions
      merged = defaultdict(list)
      prioritized(self.npcs.actions(self), merged)
      prioritized(decisions, merged)

      #Perform actions
      keys = sorted(merged)
      for priority in keys:
         actions = merged[priority] 
         for tup in actions:
            entID, atnArgs = tup
            atn, args      = atnArgs
            if entID < 0:
               entity = self.npcs[entID]
            else:
               entity = self.players[entID]

            atn.call(self, entity, *args)

      self.players.step(decisions)

      #Cull dead
      dead = self.players.cull()
      self.npcs.cull()

      self.map.step()

      #self.stats.update(self.players, self.npcs, self.market)
      self.tick += 1
      return dead

