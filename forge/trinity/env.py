from pdb import set_trace as T
import numpy as np

from collections import defaultdict
from itertools import chain
from copy import deepcopy

from forge.blade import entity, core
from forge.blade.io import stimulus
from forge.blade.io.stimulus import Static
from forge.blade.systems import combat

from forge.blade.lib.enums import Palette
from forge.blade.lib import log

class Env:
   '''Environment wrapper for Neural MMO

   Note that the contents of (ob, reward, done, info) returned by the standard
   OpenAI Gym API reset() and step(actions) methods have been generalized to
   support variable agent populations and more expressive observation/action
   spaces. This means you cannot use preexisting optimizer implementations that
   strictly expect the OpenAI Gym API. We recommend PyTorch+RLlib to take
   advantage of our prebuilt baseline implementations, but any framework that
   supports RLlib's fairly popular environment API and extended OpenAI
   gym.spaces observation/action definitions should work as well.'''
   def __init__(self, config):
      '''
      Args:
         config : A forge.blade.core.Config object or subclass object
      '''
      super().__init__()
      self.realm     = core.Realm(config, self.spawn)

      self.config    = config
      self.overlay   = None

      self.dead      = []
      self.spawned   = 0

      #self.steps = 0

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
      self.quill = log.Quill(self.realm.identify)
      
      if idx is None:
         idx = np.random.randint(self.config.TERRAIN_TRAIN_MAPS) + 1

      self.worldIdx = idx
      self.realm.reset(idx)

      obs = None
      if step:
         obs, _, _, _ = self.step({})

      return obs

   def step(self, actions, preprocess=set(), omitDead=True):
      '''OpenAI Gym API step function simulating one game tick or timestep

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

         preprocess: Per-agentID dict of whether to treat actions as raw indices
            or as game objects. Typically, this value should be True for
            neural models and false for scripted baselines.

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
            This is provided only for conformity with OpenAI Gym. Your
            algorithm should not attempt to leverage observations outside of
            trajectory bounds. You can omit garbage obs_i values by setting
            omitDead=True.

         infos:
            An empty dictionary provided only for conformity with OpenAI Gym.
      '''
      #Preprocess actions
      for entID in preprocess:
         ent = self.realm.players[entID]
         if not ent.alive:
            continue

         for atn, args in actions[entID].items():
            for arg, val in args.items():
               if len(arg.edges) > 0:
                  actions[entID][atn][arg] = arg.edges[val]
               elif val < len(ent.targets):
                  targ                     = ent.targets[val]
                  actions[entID][atn][arg] = self.realm.entity(targ)
               else: #Need to fix -inf in classifier before removing this
                  actions[entID][atn][arg] = ent

      #Step: Realm, Observations, Logs
      self.dead = self.realm.step(actions)
      obs, rewards, dones, self.raw = {}, {}, {}, {}
      for entID, ent in self.realm.players.items():
         ob             = self.realm.dataframe.get(ent)
         obs[entID]     = ob
         self.dummy_ob  = ob

         rewards[entID] = self.reward(ent)
         dones[entID]   = False

      #self.steps += len(self.realm.players.items())
      #print('World {} Tick {} Steps {}'.format(self.worldIdx, self.realm.tick, self.steps))

      for entID, ent in self.dead.items():
         self.log(ent)

      #Postprocess dead agents
      if omitDead:
         return obs, rewards, dones, {}

      for entID, ent in self.dead.items():
         rewards[ent.entID] = self.reward(ent)
         dones[ent.entID]   = True
         obs[ent.entID]     = self.dummy_ob

      return obs, rewards, dones, {}

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

   def spawn(self):
      '''Called when an agent is added to the environment

      Override this method to specify name/population upon spawning.

      Returns:
         (int, str):

         popID:
            An integer used to identity membership within a population

         prefix:
            The agent will be named prefix + entID

      Notes:
         Mainly intended for population-based research. In particular, it
         allows you to define behavior that selectively spawns agents into
         particular populations based on the current game state -- for example,
         current population sizes or performance.'''

      popSize = self.config.NENT / self.config.NPOP
      pop     = (self.spawned // popSize) % self.config.NPOP
      self.spawned += 1
      return int(pop), 'Neural_'

   ############################################################################
   ### Client data
   @property
   def packet(self):
      '''Data packet used by the renderer

      Returns:
         packet: A packet of data for the client
      '''
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

      return packet

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
               self.realm.dataframe.remove(Static.Entity, entID, ent.pos)

            observations[entID] = obs
            ents[entID] = ent
            entID += 1

      return observations, ents
