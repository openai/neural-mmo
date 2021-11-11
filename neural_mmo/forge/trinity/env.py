from pdb import set_trace as T
import numpy as np

from collections import defaultdict
from itertools import chain
from copy import deepcopy
import functools

from neural_mmo.forge.blade import entity, core
from neural_mmo.forge.blade.core.config import Config
from neural_mmo.forge.blade.io import stimulus
from neural_mmo.forge.blade.io.stimulus.static import Stimulus
from neural_mmo.forge.blade.io.action import static as Action
from neural_mmo.forge.blade.systems import combat

from neural_mmo.forge.blade.lib import log

from neural_mmo.forge.trinity.overlay import OverlayRegistry
from neural_mmo.forge.trinity.dataframe import DataType

import gym
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

class Env(ParallelEnv):
   '''Environment wrapper for Neural MMO using the Parallel PettingZoo API

   Neural MMO provides complex environments featuring structured observations/actions,
   variably sized agent populations, and long time horizons. Usage in conjunction
   with RLlib as demonstrated in the /projekt wrapper is highly recommended.

   Due to a limitation in current RLlib support for PettingZoo, this wrapper
   preallocates 2048 agents by default.'''

   metadata = {'render.modes': ['human'], 'name': 'neural-mmo'}

   def __init__(self, config):
      '''
      Args:
         config : A forge.blade.core.Config object or subclass object
      '''
      super().__init__()
      self.realm      = core.Realm(config)
      self.registry   = OverlayRegistry(config, self)

      if __debug__:
         err = 'Config {} is not a config instance (did you pass the class?)'
         assert isinstance(config, Config), err.format(config)

      self.config     = config
      self.overlay    = None
      self.overlayPos = [256, 256]
      self.client     = None
      self.obs        = None

   @functools.lru_cache(maxsize=None)
   def observation_space(self, agent: int):
      '''Neural MMO Observation Space

      Args:
         agent: Agent ID

      Returns:
         observation: gym.spaces object contained the structured observation
         for the specified agent. Each visible object is represented by
         continuous and discrete vectors of attributes. A 2-layer attentional
         encoder can be used to convert this structured observation into
         a flat vector embedding.'''

      observation = {}
      for entity in sorted(Stimulus.values()):
         rows       = entity.N(self.config)
         continuous = 0
         discrete   = 0

         for _, attr in entity:
            if attr.DISCRETE:
               discrete += 1
            if attr.CONTINUOUS:
               continuous += 1

         name = entity.__name__
         observation[name] = {
               'Continuous': gym.spaces.Box(low=-2**20, high=2**20, shape=(rows, continuous), dtype=DataType.CONTINUOUS),
               'Discrete'  : gym.spaces.Box(low=0, high=4096, shape=(rows, discrete), dtype=DataType.DISCRETE)}

         if name == 'Entity':
            observation['Entity']['N'] = gym.spaces.Box(low=0, high=self.config.N_AGENT_OBS, shape=(1,), dtype=DataType.DISCRETE)

         observation[name] = gym.spaces.Dict(observation[name])

      return gym.spaces.Dict(observation)

   @functools.lru_cache(maxsize=None)
   def action_space(self, agent):
      '''Neural MMO Action Space

      Args:
         agent: Agent ID

      Returns:
         actions: gym.spaces object contained the structured actions
         for the specified agent. Each action is parameterized by a list
         of discrete-valued arguments. These consist of both fixed, k-way
         choices (such as movement direction) and selections from the
         observation space (such as targeting)'''

      actions = {}
      for atn in sorted(Action.Action.edges):
         actions[atn] = {}
         for arg in sorted(atn.edges):
            n                       = arg.N(self.config)
            actions[atn][arg] = gym.spaces.Discrete(n)

         actions[atn] = gym.spaces.Dict(actions[atn])

      return gym.spaces.Dict(actions)
 
   ############################################################################
   ### Core API
   def reset(self, idx=None, step=True):
      '''OpenAI Gym API reset function

      Loads a new game map and returns initial observations

      Args:
         idx: Map index to load. Selects a random map by default

         step: Whether to step the environment and return initial obs

      Returns:
         obs: Initial obs if step=True, None otherwise 

      Notes:
         Neural MMO simulates a persistent world. Ideally, you should reset
         the environment only once, upon creation. In practice, this approach
         limits the number of parallel environment simulations to the number
         of CPU cores available. At small and medium hardware scale, we
         therefore recommend the standard approach of resetting after a long
         but finite horizon: ~1000 timesteps for small maps and
         5000+ timesteps for large maps

      Returns:
         observations, as documented by step()
      '''
      self.actions = {}
      self.dead    = []

      self.quill = log.Quill()
      
      if idx is not None:
         pass
      elif self.config.EVALUATE and self.config.GENERALIZE:
         idx = -np.random.randint(self.config.TERRAIN_EVAL_MAPS) - 1
      else:
         idx = np.random.randint(self.config.TERRAIN_TRAIN_MAPS) + 1

      self.worldIdx = idx
      self.realm.reset(idx)

      if step:
         self.obs, _, _, _ = self.step({})

      return self.obs

   def close(self):
       '''For conformity with the PettingZoo API only; rendering is external'''
       pass

   def step(self, actions, preprocess=set(), omitDead=True):
      '''Simulates one game tick or timestep

      Args:
         actions: A dictionary of agent decisions of format::

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

            The environment only evaluates provided actions for provided
            agents. Unprovided action types are interpreted as no-ops and
            illegal actions are ignored

            It is also possible to specify invalid combinations of valid
            actions, such as two movements or two attacks. In this case,
            one will be selected arbitrarily from each incompatible sets.

            A well-formed algorithm should do none of the above. We only
            Perform this conditional processing to make batched action
            computation easier.

         preprocess: set of agent IDs for which actions are returned as raw
            indices and need to be preprocessed. Typically this should only
            include IDs of agents controlled by neural models and exclude IDs
            of scripted agents

         omitDead: Whether to omit dead agents observations from the returned
            obs. Provided for conformity with some optimizer APIs

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
            obs_i is specified by the observation_space function.
           
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
            all other circumstances. Override Env.reward to specify
            custom reward functions
 
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
            This is provided only for conformity with PettingZoo. Your
            algorithm should not attempt to leverage observations outside of
            trajectory bounds. You can omit garbage obs_i values by setting
            omitDead=True.

         infos:
            A dictionary of agent infos of format:

               {
                  agent_1: None,
                  agent_2: None,
                  ...
               ]

            Provided for conformity with PettingZoo
      '''
      #Preprocess actions for neural models
      for entID in list(actions.keys()):
         ent = self.realm.players[entID]
         if not ent.alive:
            continue

         self.actions[entID] = {}
         for atn, args in actions[entID].items():
            self.actions[entID][atn] = {}
            for arg, val in args.items():
               if len(arg.edges) > 0:
                  self.actions[entID][atn][arg] = arg.edges[val]
               elif val < len(ent.targets):
                  targ                     = ent.targets[val]
                  self.actions[entID][atn][arg] = self.realm.entity(targ)
               else: #Need to fix -inf in classifier before removing this
                  self.actions[entID][atn][arg] = ent

      #Step: Realm, Observations, Logs
      self.dead    = self.realm.step(self.actions)
      self.actions = {}
      self.obs     = {}

      obs, rewards, dones, self.raw = {}, {}, {}, {}
      for entID, ent in self.realm.players.items():
         ob = self.realm.dataframe.get(ent)
         self.obs[entID] = ob
         if ent.agent.scripted:
            atns = ent.agent(ob)
            if Action.Attack in atns:
               atn  = atns[Action.Attack]
               targ = atn[Action.Target]
               atn[Action.Target] = self.realm.entity(targ)
            self.actions[entID] = atns
         else:
            obs[entID]     = ob
            self.dummy_ob  = ob

            rewards[entID] = self.reward(ent)
            dones[entID]   = False

      for entID, ent in self.dead.items():
         self.log(ent)

      #Postprocess dead agents
      #if omitDead:
      #   return obs, rewards, dones, {}

      for entID, ent in self.dead.items():
         if ent.agent.scripted:
            continue
         rewards[ent.entID] = self.reward(ent)
         dones[ent.entID]   = True
         obs[ent.entID]     = self.dummy_ob

      #Pettingzoo API
      self.agents = list(self.realm.players.keys())

      infos = {}
      for agent in obs:
         infos[agent] = {}

      self.obs = obs
      return obs, rewards, dones, infos

   ############################################################################
   ### Logging
   def log(self, ent) -> None:
      '''Logs agent data upon death

      This function is called automatically when an agent dies. Logs are used
      to compute summary stats and populate the dashboard. You should not
      call it manually. Instead, override this method to customize logging.

      Args:
         ent: An agent
      '''

      quill = self.quill

      blob = quill.register('Population', self.realm.tick,
            quill.HISTOGRAM, quill.LINE, quill.SCATTER)
      blob.log(self.realm.population)

      blob = quill.register('Lifetime', self.realm.tick,
            quill.HISTOGRAM, quill.LINE, quill.SCATTER, quill.GANTT)
      blob.log(ent.history.timeAlive.val)

      blob = quill.register('Skill Level', self.realm.tick,
            quill.HISTOGRAM, quill.STACKED_AREA, quill.STATS, quill.RADAR)
      blob.log(ent.skills.range.level,        'Range')
      blob.log(ent.skills.mage.level,         'Mage')
      blob.log(ent.skills.melee.level,        'Melee')
      blob.log(ent.skills.constitution.level, 'Constitution')
      blob.log(ent.skills.defense.level,      'Defense')
      blob.log(ent.skills.fishing.level,      'Fishing')
      blob.log(ent.skills.hunting.level,      'Hunting')

      blob = quill.register('Equipment', self.realm.tick,
            quill.HISTOGRAM, quill.SCATTER)
      blob.log(ent.loadout.chestplate.level, 'Chestplate')
      blob.log(ent.loadout.platelegs.level,  'Platelegs')

      blob = quill.register('Exploration', self.realm.tick,
            quill.HISTOGRAM, quill.SCATTER)
      blob.log(ent.history.exploration)

      quill.stat('Lifetime',  ent.history.timeAlive.val)

      if self.config.game_system_enabled('Achievement'):
         quill.stat('Achievement', ent.achievements.score())
         for name, stat in ent.achievements.stats:
            quill.stat(name, stat)

      if not self.config.EVALUATE:
         return

      quill.stat('PolicyID', ent.agent.policyID)

   def terminal(self):
      '''Logs currently alive agents and returns all collected logs

      Automatic log calls occur only when agents die. To evaluate agent
      performance over a fixed horizon, you will need to include logs for
      agents that are still alive at the end of that horizon. This function
      performs that logging and returns the associated a data structure
      containing logs for the entire evaluation

      Returns:
         Log datastructure. Use them to update an InkWell logger.
         
      Args:
         ent: An agent
      '''

      for entID, ent in self.realm.players.entities.items():
         self.log(ent)

      return self.quill.packet

   ############################################################################
   ### Override hooks
   def reward(self, entID):
      '''Computes the reward for the specified agent

      Override this method to create custom reward functions. You have full
      access to the environment state via self.realm. Our baselines do not
      modify this method; specify any changes when comparing to baselines

      Returns:
         float:

         reward:
            The reward for the actions on the previous timestep of the
            entity identified by entID.
      '''
      if entID not in self.realm.players:
         return -1
      return 0

   ############################################################################
   ### Client data
   def render(self, mode='human'):
      '''Data packet used by the renderer

      Returns:
         packet: A packet of data for the client
      '''
      #RLlib likes rendering for no reason
      if not self.config.RENDER:
         return 

      packet = {
            'config': self.config,
            'pos': self.overlayPos,
            'wilderness': 0
            }

      packet = {**self.realm.packet(), **packet}

      if self.overlay is not None:
         print('Overlay data: ', len(self.overlay))
         packet['overlay'] = self.overlay
         self.overlay      = None

      if not self.client:
         from neural_mmo.forge.trinity.twistedserver import Application
         self.client = Application(self) 

      pos, cmd = self.client.update(packet)
      if self.obs:
         self.registry.step(self.obs, pos, cmd)

   def register(self, overlay):
      '''Register an overlay to be sent to the client

      The intended use of this function is: User types overlay ->
      client sends cmd to server -> server computes overlay update -> 
      register(overlay) -> overlay is sent to client -> overlay rendered

      Args:
         values: A map-sized (self.size) array of floating point values
      '''
      err = 'overlay must be a numpy array of dimension (*(env.size), 3)'
      assert type(overlay) == np.ndarray, err
      self.overlay = overlay.tolist()

   def dense(self):
      '''Simulates an agent on every tile and returns observations

      This method is used to compute per-tile visualizations across the
      entire map simultaneously. To do so, we spawn agents on each tile
      one at a time. We compute the observation for each agent, delete that
      agent, and go on to the next one. In this fashion, each agent receives
      an observation where it is the only agent alive. This allows us to
      isolate potential influences from observations of nearby agents

      This function is slow, and anything you do with it is probably slower.
      As a concrete example, consider that we would like to visualize a
      learned agent value function for the entire map. This would require
      computing a forward pass for one agent per tile. To cut down on
      computation costs, we omit lava tiles from this method

      Returns:
         (dict, dict):

         observations:
            A dictionary of agent observations as specified by step()

         ents:
            A corresponding dictionary of agents keyed by their entID
      '''
      config  = self.config
      R, C    = self.realm.map.tiles.shape

      entID   = 100000
      pop     = 0
      name    = "Value"
      color   = (255, 255, 255)


      observations, ents = {}, {}
      for r in range(R):
         for c in range(C):
            tile    = self.realm.map.tiles[r, c]
            if not tile.habitable:
               continue

            current = tile.ents
            n       = len(current)
            if n == 0:
               ent = entity.Player(self.realm, (r, c), entID, pop, name, color)
            else:
               ent = list(current.values())[0]

            obs = self.realm.dataframe.get(ent)
            if n == 0:
               self.realm.dataframe.remove(Stimulus.Entity, entID, ent.pos)

            observations[entID] = obs
            ents[entID] = ent
            entID += 1

      return observations, ents
