from pdb import set_trace as T
import numpy as np
import time

from collections import defaultdict
from itertools import chain
from copy import deepcopy

from forge.blade import entity, core
from forge.blade.io import stimulus
from forge.blade.io.stimulus import Static
from forge.blade.systems import combat

from forge.blade.lib.enums import Palette
from forge.blade.lib import log
from forge.trinity.ascend import runtime, Timed

def valToRGB(x):
    '''x in range [0, 1]'''
    return np.array([1-x, x, 0])

class Env(Timed):
   '''Neural MMO environment class implementing the OpenAI Gym API function
   signatures. The actual (ob, reward, done, info) data contents returned by
   the canonical reset() and step(action) methods conform to RLlib's Gym
   extensions in order to support multiple and variably sized agent
   populations. This means you cannot use preexisting optimizer
   implementations that expect the OpenAI Gym API. We recommend
   PyTorch+RLlib to take advantage of our prebuilt baseline implementations,
   but any framework that supports RLlib's fairly popular environment API and
   extended OpenAI gym.spaces observation/action definitions works as well.'''
   def __init__(self, config, idx=0):
      '''
      Args:
         config : A forge.blade.core.Config (or subclass) specification object
         idx    : Index of the map file to load (0 to number of maps)
      '''
      super().__init__()
      self.realm     = core.Realm(config, idx)

      self.config    = config
      self.worldIdx  = idx
      self.overlay   = None

      self.entID     = 1
      self.tick      = 0

   @runtime
   def step(self, decisions):
      '''OpenAI Gym API step function simulating one game tick or timestep

      Args:
         decisions: A dictionary of agent action choices of format::

               {
                  agent_1: {
                     action_1: [arg_1, arg_2],
                     action_2: [...],
                     ...
                  },
                  agent_2: {
                     ...
                  },
                  ...
               }

            Where agent_i is the integer index of the i\'th agent 

            You do not need to provide actions for each agent
            You do not need to provide an action of each type for each agent
            Only provided actions for provided agents will be evaluated
            Unprovided action types are interpreted as no-ops
            Invalid actions are ignored

            It is also possible to specify invalid combinations of valid
            actions, such as two movements or two attacks. In this case,
            one will be selected arbitrarily from each incompatible sets.

            A well-formed algorithm should do none of the above. We only
            Perform this conditional processing to make batched action
            computation easier.

      Returns:
         (dict, dict, dict, None):

         observations:
            A dictionary of agent observations of format::

               {
                  agent_1: obs_1,
                  agent_2: obs_2,
                  ...
               ]

            Where agent_i is the integer index of the i\'th agent and
            obs_i is the observation of the i\'th' agent. Note that obs_i
            is a structured datatype -- not a flat tensor. It is automatically
            interpretable under an extended OpenAI gym.spaces API. Our demo
            code shows how do to this in RLlib. Other frameworks must
            implement the same extended gym.spaces API to do the same.
            
         rewards:
            A dictionary of agent rewards of format::

               {
                  agent_1: reward_1,
                  agent_2: reward_2,
                  ...
               ]

            Where agent_i is the integer index of the i\'th agent and
            reward_i is the reward of the i\'th' agent.

            By default, agents receive -1 reward for dying and 0 reward for
            all other circumstances. Realm.hook provides an interface for
            creating custom reward functions using full game state.
 
         dones:
            A dictionary of agent done booleans of format::

               {
                  agent_1: done_1,
                  agent_2: done_2,
                  ...
               ]

            Where agent_i is the integer index of the i\'th agent and
            done_i is a boolean denoting whether the i\'th agent has died.

            
            Note that obs_i will be a garbage placeholder if done_i is true.
            This is provided only for conformity with OpenAI Gym. Your
            algorithm should not attempt to leverage observations outside of
            trajectory bounds.

         infos:
            An empty dictionary provided only for conformity with OpenAI Gym.
      '''
      self.tick += 1

      self.realm_step = time.time()

      #Spawn an ent
      iden, pop, name = self.spawn()
      assert iden is not None

      self.realm.players.spawn(iden, pop, name)
      self.dead = self.realm.step(decisions)

      self.realm_step = time.time() - self.realm_step
      self.env_stim   = time.time()

      obs, rewards, dones = self.getStims(self.dead)
      self.env_stim       = time.time() - self.env_stim

      infos = {}
      if self.config.RENDER:
         infos = log.Quill(self.worldIdx, self.tick)
         for ent in self.dead:
            self.log(infos, ent)
         infos = infos.packet

      return obs, rewards, dones, infos

   def log(self, blob, ent):
      pass

   def terminal(self):
      infos = log.Quill(self.worldIdx, self.tick)
      for entID, ent in self.realm.players.entities.items():
         self.log(infos, ent)

      return infos.packet

   def reset(self):
      '''Instantiates the environment and returns initial observations

      Neural MMO simulates a persistent world. It is best-practice to call
      reset() once per environment upon initialization and never again.
      Treating agent lifetimes as episodes enables training with all on-policy
      and off-policy reinforcement learning algorithms.

      We provide this function for conformity with OpenAI Gym and
      compatibility with various existing off-the-shelf reinforcement
      learning algorithms that expect a hard environment reset. If you
      absolutely must call this method after the first initialization,
      we suggest using very long (1000+) timestep environment simulations.

      Returns:
         observations, as documented by step()
      '''
      err = 'Neural MMO is persistent and may only be reset once upon initialization'
      assert self.tick == 0, err
      return self.step({})[0]

   def reward(self, entID):
      '''Computes the reward for the specified agent

      You can override this method to create custom reward functions.
      This method has access to the full environment state via self.
      The baselines do not modify this method. You should specify any
      changes you may have made to this method when comparing to the baselines

      Returns:
         float:

         reward:
            The reward for the actions on the previous timestep of the
            entity identified by entID.
      '''
      if entID not in self.realm.players:
         return -1
      return 0

   def spawn(self):
      '''Called when an agent is added to the environment

      You can override this method to specify custom spawning behavior
      with full access to the environment state via self.

      Returns:
         (int, int, str):

         entID:
            An integer used to uniquely identify the entity

         popID:
            An integer used to identity membership within a population

         prefix:
            The agent will be named prefix + entID

      Notes:
         This API hook is mainly intended for population-based research. In
         particular, it allows you to define behavior that selectively
         spawns agents into particular populations based on the current game
         state -- for example, current population sizes or performance.'''
      pop        =  hash(str(self.entID)) % self.config.NPOP
      self.entID += 1

      return self.entID, pop, 'Neural_'

   def clientData(self):
      '''Data packet used by the renderer

      Returns:
         packet: A packet of data for the client
      '''
      packet = {
            'config': self.config,
            'environment': self.realm.map,
            'resource': self.realm.map.packet(),
            'player': dict((k, v.packet())
               for k, v in self.realm.players.items()),
            'npc': dict((k, v.packet())
               for k, v in self.realm.npcs.items()),
            'pos': self.overlayPos,
            'wilderness': combat.wilderness(self.config, self.overlayPos)
            }

      if self.overlay is not None:
         print('Overlay data: ', len(self.overlay))
         packet['overlay'] = self.overlay
         self.overlay      = None

      return packet

   def getStims(self, dead):
      '''Gets agent stimuli from the environment

      Args:
         packets: A dictionary of Packet objects

      Returns:
         The packet dictionary populated with agent data
      '''
      self.stim_process = 0

      self.raw = {}
      obs, rewards, dones = {}, {}, {'__all__': False}
      for entID, ent in self.realm.players.items():
         start = time.time()
         ob             = self.realm.dataframe.get(ent)
         obs[entID]     = ob
         self.stim_process += time.time() - start

         rewards[entID] = self.reward(entID)
         dones[entID]   = False

      for ent in dead:
         #Why do we have to provide an ob for the last timestep?
         #Currently just copying one over
         rewards[ent.entID] = self.reward(ent)
         dones[ent.entID]   = True
         obs[ent.entID]     = ob

      return obs, rewards, dones

   @property
   def size(self):
      '''Returns the size of the game map

      You can override this method to create custom reward functions.
      This method has access to the full environment state via self.
      The baselines do not modify this method. You should specify any
      changes you may have made to this method when comparing to the baselines

      Returns:
         tuple(int, int):

         size:
            The size of the map as (rows, columns)
      '''
      return self.realm.map.tiles.shape

   def registerOverlay(self, overlay):
      '''Registers an overlay to be sent to the client

      This variable is included in client data passed to the renderer and is
      typically used to send value maps computed using getValStim to the
      client in order to render as an overlay.

      Args:
         values: A map-sized (self.size) array of floating point values
      '''
      err = 'overlay must be a numpy array of dimension (*(env.size), 3)'
      assert type(overlay) == np.ndarray, err
      self.overlay = overlay.tolist()

   def getValStim(self):
      '''Simulates an agent on every tile and returns observations

      This method is used to compute per-tile visualizations across the
      entire map simultaneously. To do so, we spawn agents on each tile
      one at a time. We compute the observation for each agent, delete that
      agent, and go on to the next one. In this fashion, each agent receives
      an observation where it is the only agent alive. This allows us to
      isolate potential influences from observations of nearby agents

      This function is slow, and anything you do with it is porbably slower.
      As a concrete example, consider that we would like to visualize a
      learned agent value function for the entire map. This would require
      computing a forward pass for one agent per tile. To cut down on
      computation costs, we omit lava tiles from this method

      Returns:
         (dict, dict):

         observations:
            A dictionary of agent observations as specified by step()

         stimuli:
            A dictionary of raw game object observations as follows::

               {
                  agent_1: (tiles, agent),
                  agent_2: (tiles, agent),
                  ...
               ]

            Where agent_i is the integer index of the i\'th agent,
            tiles is an array of observed game tiles, and agent is the
            game object corresponding to agent_i
      '''
      config  = self.config
      R, C    = self.realm.map.tiles.shape
      B       = config.TERRAIN_BORDER

      entID   = 0
      pop     = 0
      name    = "Value"
      color   = (255, 255, 255)

      observations, ents = {}, {}
      for r in range(B-1, R-B):
         for c in range(B-1, C-B):
            ent = entity.Player(self.realm, (r, c), entID, pop, name, color)

            self.realm.map.tiles[r, c].addEnt(entID, ent)
            obs = self.realm.dataframe.get(ent)

            self.realm.map.tiles[r, c].delEnt(entID)
            self.realm.dataframe.remove(Static.Entity, entID, ent.pos)

            observations[entID] = obs
            ents[entID] = ent
            entID += 1

      return observations, ents
