from pdb import set_trace as T

from collections import defaultdict
from itertools import chain

from forge.blade import entity, core
from forge.blade.lib.enums import Palette
from forge.trinity.ascend import runtime, Timed

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
      realm.world.env.tiles[r, c].counts[ent.base.population.val] += 1

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
   '''Neural MMO Environment'''
   def __init__(self, config, idx=0):
      '''
      Args:

         config : A Config specification object
         args   : Hook for command line arguments
         idx    : Index of the map file to load
      '''
      super().__init__()
      self.spawner   = Spawner(config)
      self.world     = core.Env(config, idx)
      self.env       = self.world.env

      self.config    = config
      self.worldIdx  = idx
      self.desciples = {}

      self.entID     = 1
      self.tick      = 0

   def clientData(self):
      '''Data packet used by the renderer

      Returns:
         packet: A packet of data for the client
      '''
      packet = {
            'environment': self.world.env,
            'entities': dict((k, v.packet()) 
               for k, v in self.desciples.items()),
            'values': []
            }

      return packet

   def spawn(self):
      '''Specifies the environment protocol for adding players
      Override to specify custom spawning behavior

      Returns:
         entID  : A unique entity ID
         pop    : A population membership ID
         prefix : A string prepended to agent names

      Notes:
         This fomulation is useful for population based research, as it
         allows one to specify per-agent or per-population policies'''

      pop        =  hash(str(self.entID)) % self.config.NPOP
      self.entID += 1

      return self.entID, pop, 'Neural_'

   def reward(self, entID):
      '''Specifies the environment protocol for rewarding agents
      Override to specify custom reward behavior

      Returns:
         reward: floating point reward value

      Notes:
         Default behavior returns only zero. You will need to interpret
         the "done" signal as a -1 during rollout collection'''

      return 0
 
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
      dead, dones = set(), set()
      packets     = defaultdict(Packet)
      for entID in decisions.keys():
         ent    = self.desciples[entID]
         if self.postmortem(ent, dead):
            dones.add(ent.serial)
         else:
            packets[entID].reward = self.reward(entID)
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
         dead.add(entID)
         return True
      return False

   def cullDead(self, dead):
      '''Deletes the specified list of agents

      Args: 
         dead: A list of dead agent IDs to remove
      '''
      for entID in dead:
         ent  = self.desciples[entID]
         r, c = ent.base.pos

         self.world.env.tiles[r, c].delEnt(entID)
         self.spawner.cull(ent.annID)

         del self.desciples[entID]

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
         stim = self.world.env.stim(
                ent.base.pos, self.config.STIM)

         packets[entID].stim = (stim, ent)

      return packets

   @runtime
   def step(self, decisions):
      '''Take actions for all agents and return new observations

      Args:
         decisions: A dictionary of agent decisions

      Returns:
         observations : A list of local game state observations of form (env, ent) for each agent where "env" is a grid of game tile objects and "ent" is a game entity object representing the current agent. Use forge.io libraries for preprocessing.
         reward       : A list of rewards for each agent (floating point or None)
         dones        : A set of IDs corresponding to agents that have died during the past game tick
         info         : None -- provided for conformity with OpenAI Gym
      '''
      self.tick += 1

      #Spawn an ent
      iden, pop, name = self.spawn()
      assert iden is not None

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
      err = 'Neural MMO is persistent and may only be reset once upon initialization'
      assert self.tick == 0, err
      return self.step({})


