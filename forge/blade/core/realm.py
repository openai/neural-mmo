from pdb import set_trace as T
import numpy as np

from collections import defaultdict
from itertools import chain
from copy import deepcopy

from forge.blade import entity, core
from forge.blade.io import stimulus

from forge.blade.lib.enums import Palette
from forge.trinity.ascend import runtime, Timed

def valToRGB(x):
    '''x in range [0, 1]'''
    return np.array([1-x, x, 0])

class Packet():
   '''Wrapper for state, reward, done signals'''
   def __init__(self):
      '''Instantiates packet data'''
      self.stim   = None
      self.reward = None
      self.done   = None

class Spawner:
   '''Manager class responsible for agent spawning logic'''
   def __init__(self, config):
      '''
      Args:
         config: A Config object
      '''
      self.nEnt, self.nPop = config.NENT, config.NPOP
      self.popSz  = self.nEnt // self.nPop
      self.config = config

      self.ents = 0
      self.pops = defaultdict(int)
      self.palette = Palette(self.nPop)

   def spawn(self, realm, iden, pop, name):
      '''Adds an entity to the given environment

      Args:
         realm : An environment Realm object
         iden  : An identifier index to assign to the new agent
         pop   : A population index to assign to the new agent
         name  : A string to prepend to iden as an agent name
      '''

      assert iden is not None
      assert self.pops[pop] <= self.popSz
      assert self.ents      <= self.nEnt

      #Too many total entities
      if self.ents == self.nEnt:
         return None

      if self.pops[pop] == self.popSz:
         return None

      self.pops[pop] += 1
      self.ents      += 1

      color = self.palette.colors[pop]
      ent   = entity.Player(self.config, iden, pop, name, color)
      assert ent not in realm.desciples

      r, c = ent.base.pos
      realm.desciples[ent.entID] = ent
      realm.world.env.tiles[r, c].addEnt(iden, ent)
      realm.world.env.tiles[r, c].count.value += 1
      realm.world.env.tiles[r, c].count._color += np.array(ent.base.color.rgb)

   def cull(self, pop):
      '''Decrement the agent counter for the specified population

      Args: 
         pop: A population index
      '''
      assert self.pops[pop] >= 1
      assert self.ents      >= 1

      self.pops[pop] -= 1
      self.ents      -= 1

      if self.pops[pop] == 0:
         del self.pops[pop]

class Realm(Timed):
   '''This module is the core Neural MMO environment and only 
   documents the internal API. External documentation is available
   at :mod:`forge.blade.core.ascend`'''
   def __init__(self, config, idx=0):
      super().__init__()
      self.spawner   = Spawner(config)
      self.world     = core.Env(config, idx)
      self.env       = self.world.env

      self.globalValues = None
      self.config    = config
      self.worldIdx  = idx
      self.desciples = {}

      self.entID     = 1
      self.tick      = 0

   ###################################################################
   ### Internal API (+ constructor) documented at forge.blade.core.api
   ### This is mainly to avoid dealing with decorated method doc gen

   @runtime
   def step(self, decisions, values={}, attn={}):
      self.tick += 1

      #Overlay data
      for entID, val in values.items():
         r, c = self.desciples[entID].base.pos
         self.world.env.tiles[r, c].value.value = val

      for tile in self.world.env.tiles.ravel():
          tile.attention.value = 0
          tile.attention.updates = 0
          tile.attention._color *= 0

      for entity, gameObjs in attn.items():
         rgb = np.array(entity.base.color.rgb) / 255
         for obj, val in gameObjs.items():
             if type(obj).__name__ == 'Tile':
                r, c = obj.r, obj.c
                #rgb = valToRGB(np.clip(val, 0, 1))
                self.world.env.tiles[r, c].attention.value  += val
                self.world.env.tiles[r, c].attention._color += rgb * val

      for tile in self.world.env.tiles.ravel():
          if tile.attention.updates > 0:
             tile.attention.value /= tile.attention.updates

      #Spawn an ent
      iden, pop, name = self.spawn()
      assert iden is not None

      self.spawner.spawn(self, iden, pop, name)
      self.stepEnts(decisions)

      self.stepEnv()
      obs, rewards, dones = self.getStims()

      return obs, rewards, dones, {}

   def reset(self):
      err = 'Neural MMO is persistent and may only be reset once upon initialization'
      assert self.tick == 0, err
      return self.step({})

   def reward(self, entID):
      if entID in self.dead:
         return -1
      return 0

   def spawn(self):
      pop        =  hash(str(self.entID)) % self.config.NPOP
      self.entID += 1

      return self.entID, pop, 'Neural_'

   ### End Internal API
   ###################################################################

   def clientData(self):
      '''Data packet used by the renderer

      Returns:
         packet: A packet of data for the client
      '''
      packet = {
            'environment': self.world.env,
            'entities': dict((k, v.packet()) 
               for k, v in self.desciples.items()),
            'globalValues': self.globalValues
            }

      #np.random.rand(80, 80, 3),
      return packet

   def act(self, actions):
      '''Execute agent actions
      
      Args:
         actions: A dictionary of agent actions
      '''
      for priority, tups in actions.items():
         for entID, atnArgs in tups.items():
            ent = self.desciples[entID]
            ent.act(self.world, atnArgs)

   def prioritize(self, decisions):
      '''Reorders actions according to their priorities

      Args:
         decisions: A dictionary of agent actions
      
      Returns:
         Repriotized actions

      Notes:
         Only saves the first action of each priority
      '''
      actions = defaultdict(dict)
      for entID, atns in decisions.items():
         for atn, args in atns.items():
            actions[atn.priority][entID] = [atn, args]
      return actions

   def stepEnv(self):
      '''Advances the environment'''
      ents = list(chain(self.desciples.values()))

      #Stats
      self.world.step(ents, [])
      self.world.env.step()

      self.env = self.world.env.np()

   def stepEnts(self, decisions):
      '''Advance agents
      
      Args:
         decisions: A dictionary of agent actions

      Returns:
         packets : State-reward-done packets
         dones   : A list of dead agent IDs
      '''
      #Step all ents first
      for entID, actions in decisions.items():
         ent = self.desciples[entID]
         ent.step(self.world, actions)

      actions = self.prioritize(decisions)
      self.act(actions)

      #Finally cull dead. This will enable MAD melee
      dead = set()
      for entID, ent in self.desciples.items():
         self.postmortem(ent, dead)

      self.dead = dead
      self.cullDead(dead)
      return dead

   def postmortem(self, ent, dead):
      '''Add agent to the graveyard if it is dead

      Args:
         ent  : An agent object
         dead : A list of dead agents

      Returns:
         bool: Whether the agent is dead
      '''
      if not ent.base.alive:
         dead.add(ent)
         return True
      return False

   def cullDead(self, dead):
      '''Deletes the specified list of agents

      Args: 
         dead: A list of dead agent IDs to remove
      '''
      for ent in dead:
         r, c  = ent.base.pos
         entID = ent.entID

         self.world.env.tiles[r, c].delEnt(entID)
         self.spawner.cull(ent.annID)

         del self.desciples[entID]

   def getStims(self):
      '''Gets agent stimuli from the environment

      Args:
         packets: A dictionary of Packet objects

      Returns:
         The packet dictionary populated with agent data
      '''
      self.raw = {}
      obs, rewards, dones = {}, {}, {'__all__': False}
      for entID, ent in self.desciples.items():
         r, c = ent.base.pos
         tile = self.world.env.tiles[r, c].tex
         stim = self.world.env.stim(
                ent.base.pos, self.config.STIM)

         obs[entID], self.raw[entID] = stimulus.Dynamic.process(
               self.config, stim, ent)
         ob = obs[entID]

         rewards[entID] = self.reward(entID)
         dones[entID]   = False

      for ent in self.dead:
         #Why do we have to provide an ob for the last timestep?
         #Currently just copying one over
         rewards[ent.entID] = self.reward(ent)
         dones[ent.entID]   = True
         obs[ent.entID]     = ob

      return obs, rewards, dones

   def setGlobalValues(self, values):
      self.globalValues = values

   def getValStim(self):
      config  = self.config
      R, C    = self.world.env.tiles.shape
      B       = config.BORDER

      entID   = 1
      pop     = 0
      name    = "Value"
      color   = (255, 255, 255)

      stims   = []
      for r in range(B-1, R-B):
         for c in range(B-1, C-B):
            pos   = tuple([r, c])
            ent   = entity.Player(config, entID, pop, name, color)

            ent.base.r.update(r)
            ent.base.c.update(c)

            self.world.env.tiles[r, c].addEnt(entID, ent)
            stim = self.world.env.stim(pos, self.config.STIM)
            stim = deepcopy(stim)
            self.world.env.tiles[r, c].delEnt(entID)

            obs, _ = stimulus.Dynamic.process(self.config, stim, ent)
            stims.append((obs, (stim, ent)))
            entID += 1
            
      return stims
