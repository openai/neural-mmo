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
from forge.trinity.ascend import runtime, Timed

class Packet():
   '''Wrapper for state, reward, done signals'''
   def __init__(self):
      '''Instantiates packet data'''
      self.stim  = None
      self.reward = None
      self.done   = None

class Spawner:
   '''Manager class responsible for agent spawning logic'''
   def __init__(self, config):
      '''
      Args:
         config: A Config object
      '''
      self.config = config

      self.nEnt, self.nPop = config.NENT, config.NPOP
      self.popSz = self.nEnt // self.nPop

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

      r, c = ent.base.pos
      realm.desciples[ent.entID] = ent
      realm.world.env.tiles[r, c].addEnt(iden, ent)
      realm.world.env.tiles[r, c].counts[ent.base.population.val] += 1

   def cull(self, pop):
      '''Decrement the agent counter for the specified population

      Args: 
         pop: A population index
      '''
      assert self.pops[pop] >= 1
      assert self.ents      >= 1

      self.pops[pop] -= 1
      self.ents -= 1

      if self.pops[pop] == 0:
         del self.pops[pop]

class Realm(Timed):
   '''Neural MMO Environment'''
   def __init__(self, config, idx, spawn):
      '''
      Args:

         config : A Config specification object
         args   : Hook for command line arguments
         idx    : Index of the map file to load
      '''
      super().__init__()
      self.spawner = Spawner(config)
      self.spawn   = spawn

      self.world, self.desciples = core.Env(config, idx), {}
      self.config = config
      self.npop = config.NPOP

      self.worldIdx = idx
      self.tick = 0

      self.env = self.world.env
      self.values = None

   def clientData(self):
      '''Data packet used by the renderer

      Returns:
         packet: A packet of data for the client
      '''
      if self.values is None:# and hasattr(self, 'sword'):
         #self.values = self.sword.anns[0].visVals()
         self.values = []

      ret = {
            'environment': self.world.env,
            'entities': dict((k, v.packet()) 
               for k, v in self.desciples.items()),
            'values': self.values
            }
      return ret

   def cullDead(self, dead):
      '''Deletes the specified list of agents

      Args: 
         dead: A list of dead agent IDs to remove
      '''
      for entID in dead:
         ent = self.desciples[entID]
         r, c = ent.base.pos
         self.world.env.tiles[r, c].delEnt(entID)
         self.spawner.cull(ent.annID)
         del self.desciples[entID]

   def stepEnv(self):
      '''Advances the environment'''
      ents = list(chain(self.desciples.values()))

      #Stats
      self.world.step(ents, [])
      self.world.env.step()

      self.env = self.world.env.np()

   def getStim(self, ent):
      '''Gets agent stimuli from the environment

      Args:
         ent: An agent object

      Returns:
         Stimuli for the given agent
      '''
      return self.world.env.stim(ent.base.pos, self.config.STIM)

   #Only saves the first action of each priority
   def prioritize(self, decisions):
      '''Reorders actions according to their priorities

      Args:
         decisions: A dictionary of agent actions
      
      Returns:
         Repriotized actions
      '''
      actions = defaultdict(dict)
      for entID, atns in decisions.items():
         for atn, args in atns.items():
            actions[atn.priority][entID] = [atn, args]
      return actions

   def act(self, actions):
      '''Execute agent actions
      
      Args:
         actions: A dictionary of agent actions
      '''
      for priority, tups in actions.items():
         for entID, atnArgs in tups.items():
            ent = self.desciples[entID]
            ent.act(self.world, atnArgs)

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
      dead  = []
      dones = []
      packets = defaultdict(Packet)
      for entID in decisions.keys():
         ent = self.desciples[entID]
         if self.postmortem(ent, dead):
            #This line was commented before
            #packets[entID].reward = -1
            dones.append(ent.serial)
         else:
            #packets[entID].reward = 0.05 * min(
            #      ent.resources.health.val,
            #      ent.resources.water.val,
            #      ent.resources.food.val)
            packets[entID].reward = 0
            packets[entID].stim = ent

      self.cullDead(dead)
      return packets, dones

   def postmortem(self, ent, dead):
      '''Add agent to the graveyard if it is dead

      Args:
         ent  : An agent object
         dead : A list of dead agents

      Returns:
         bool: Whether the agent is dead
      '''
      entID = ent.entID
      if not ent.base.alive:
         dead.append(entID)
         return True
      return False

   def getStims(self, packets):
      '''Gets agent stimuli from the environment

      Args:
         packets: A dictionary of Packet objects

      Returns:
         The packet dictionary populated with agent data
      '''
      for entID, ent in self.desciples.items():
         r, c = ent.base.pos
         tile = self.world.env.tiles[r, c].tex
         stim = self.getStim(ent)
         packets[entID].stim = (stim, ent)

      return packets

   @runtime
   def step(self, decisions):
      '''Take actions for all agents and return new observations

      Args:
         decisions: A dictionary of agent decisions

      Returns:
         observations : A dictionary of local game state
         reward       : 0
         dones        : (bool) Whether the agent has died
         info         : None
      '''
      self.tick += 1

      #Spawn an ent
      iden, pop, name = self.spawn()
      if iden is not None:
         self.spawner.spawn(self, iden, pop, name)

      packets, dead = self.stepEnts(decisions)

      self.stepEnv()
      packets = self.getStims(packets)

      #Conform to gym
      stims   = [p.stim   for p in packets.values()]
      rewards = [p.reward for p in packets.values()]
      dones   = dead

      return stims, rewards, dones, None

   def reset(self):
      '''Stub for conformity with Gym. Calls step({}).

      The environment is persistent. Reset it only
      once upon initialization to obtain initial
      observations. If you must experiment with
      short lived environment instances, instantiate
      a new Realm instead of calling reset.

      Returns:
         data: Output of self.step({})
      '''
      assert self.tick == 0
      return self.step({})


