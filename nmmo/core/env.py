from pdb import set_trace as T
import numpy as np
import random

import functools
from collections import defaultdict

import gym
from pettingzoo import ParallelEnv

import json
import lzma

import nmmo
from nmmo import entity, core, emulation
from nmmo.core import terrain
from nmmo.lib import log
from nmmo.infrastructure import DataType
from nmmo.systems import item as Item


class Replay:
    def __init__(self, config):
        self.packets = []
        self.map     = None

        if config is not None:
            self.path = config.SAVE_REPLAY + '.lzma'

        self._i = 0

    def update(self, packet):
        data = {}
        for key, val in packet.items():
            if key == 'environment':
                self.map = val
                continue
            if key == 'config':
                continue

            data[key] = val

        self.packets.append(data)

    def save(self):
        print(f'Saving replay to {self.path} ...')

        data = {
            'map': self.map,
            'packets': self.packets}

        data = json.dumps(data).encode('utf8')
        data = lzma.compress(data, format=lzma.FORMAT_ALONE)
        with open(self.path, 'wb') as out:
            out.write(data)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as fp:
            data = fp.read()

        data = lzma.decompress(data, format=lzma.FORMAT_ALONE)
        data = json.loads(data.decode('utf-8'))

        replay = Replay(None)
        replay.map = data['map']
        replay.packets = data['packets']
        return replay

    def render(self):
        from nmmo.websocket import Application
        client = Application(realm=None)
        for packet in self:
            client.update(packet)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i >= len(self.packets):
            raise StopIteration
        packet = self.packets[self._i]
        packet['environment'] = self.map
        self._i += 1
        return packet


class Env(ParallelEnv):
   '''Environment wrapper for Neural MMO using the Parallel PettingZoo API

   Neural MMO provides complex environments featuring structured observations/actions,
   variably sized agent populations, and long time horizons. Usage in conjunction
   with RLlib as demonstrated in the /projekt wrapper is highly recommended.'''

   metadata = {'render.modes': ['human'], 'name': 'neural-mmo'}

   def __init__(self, config=None):
      '''
      Args:
         config : A forge.blade.core.Config object or subclass object
      '''
      super().__init__()

      if config is None:
          config = nmmo.config.Default()

      assert isinstance(config, nmmo.config.Config), f'Config {config} is not a config instance (did you pass the class?)'

      if not config.PLAYERS:
          from nmmo import agent
          config.PLAYERS = [agent.Random]

      if not config.MAP_GENERATOR:
          config.MAP_GENERATOR = terrain.MapGenerator

      self.realm      = core.Realm(config)
      self.registry   = nmmo.OverlayRegistry(config, self)
    
      self.config     = config
      self.overlay    = None
      self.overlayPos = [256, 256]
      self.client     = None
      self.obs        = None

      self.has_reset  = False

      # Populate dummy ob
      self.dummy_ob   = None
      self.observation_space(0)

      if self.config.SAVE_REPLAY:
         self.replay = Replay(config)

      if config.EMULATE_CONST_PLAYER_N:
         self.possible_agents = [i for i in range(1, config.PLAYER_N + 1)]

      # Flat index actions
      if config.EMULATE_FLAT_ATN:
         self.flat_actions = emulation.pack_atn_space(config)

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
      for entity in sorted(nmmo.Serialized.values()):
         if not entity.enabled(self.config):
            continue

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
               'Continuous': gym.spaces.Box(
                        low=-2**20, high=2**20,
                        shape=(rows, continuous),
                        dtype=DataType.CONTINUOUS),
               'Discrete'  : gym.spaces.Box(
                        low=0, high=4096,
                        shape=(rows, discrete),
                        dtype=DataType.DISCRETE)}

         #TODO: Find a way to automate this
         if name == 'Entity':
            observation['Entity']['N'] = gym.spaces.Box(
                    low=0, high=self.config.PLAYER_N_OBS,
                    shape=(1,), dtype=DataType.DISCRETE)
         elif name == 'Tile':
            observation['Tile']['N'] = gym.spaces.Box(
                    low=0, high=self.config.PLAYER_VISION_DIAMETER,
                    shape=(1,), dtype=DataType.DISCRETE)
         elif name == 'Item':
            observation['Item']['N']   = gym.spaces.Box(low=0, high=self.config.ITEM_N_OBS, shape=(1,), dtype=DataType.DISCRETE)
         elif name == 'Market':
            observation['Market']['N'] = gym.spaces.Box(low=0, high=self.config.EXCHANGE_N_OBS, shape=(1,), dtype=DataType.DISCRETE)

         observation[name] = gym.spaces.Dict(observation[name])

      observation   = gym.spaces.Dict(observation)

      if not self.dummy_ob:
         self.dummy_ob = observation.sample()
         for ent_key, ent_val in self.dummy_ob.items():
             for attr_key, attr_val in ent_val.items():
                 self.dummy_ob[ent_key][attr_key] *= 0                


      if not self.config.EMULATE_FLAT_OBS:
         return observation

      return emulation.pack_obs_space(observation)

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

      if self.config.EMULATE_FLAT_ATN:
         lens = []
         for atn in nmmo.Action.edges(self.config):
             for arg in atn.edges:
                 lens.append(arg.N(self.config))
         return gym.spaces.MultiDiscrete(lens)
         #return gym.spaces.Discrete(len(self.flat_actions))

      actions = {}
      for atn in sorted(nmmo.Action.edges(self.config)):
         actions[atn] = {}
         for arg in sorted(atn.edges):
            n                 = arg.N(self.config)
            actions[atn][arg] = gym.spaces.Discrete(n)

         actions[atn] = gym.spaces.Dict(actions[atn])

      return gym.spaces.Dict(actions)

   ############################################################################
   ### Core API
   def reset(self, idx=None, step=True, seed=None):
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
      if seed is not None:
          np.random.seed(seed)
          random.seed(seed)

      self.has_reset = True

      self.actions = {}
      self.dead    = []

      if idx is None:
         idx = np.random.randint(self.config.MAP_N) + 1

      self.worldIdx = idx
      self.realm.reset(idx)

      # Set up logs
      self.register_logs()
 
      if step:
         self.obs, _, _, _ = self.step({})

      return self.obs

   def close(self):
       '''For conformity with the PettingZoo API only; rendering is external'''
       pass

   def step(self, actions):
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

      Returns:
         (dict, dict, dict, None):

         observations:
            A dictionary of agent observations of format::

               {
                  agent_1: obs_1,
                  agent_2: obs_2,
                  ...
               }

            Where agent_i is the integer index of the i\'th agent and
            obs_i is specified by the observation_space function.
           
         rewards:
            A dictionary of agent rewards of format::

               {
                  agent_1: reward_1,
                  agent_2: reward_2,
                  ...
               }

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
               }

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
               }

            Provided for conformity with PettingZoo
      '''
      assert self.has_reset, 'step before reset'

      if self.config.RENDER or self.config.SAVE_REPLAY:
          packet = {
                'config': self.config,
                'pos': self.overlayPos,
                'wilderness': 0
                }

          packet = {**self.realm.packet(), **packet}

          if self.overlay is not None:
             packet['overlay'] = self.overlay
             self.overlay      = None

          self.packet = packet

          if self.config.SAVE_REPLAY:
              self.replay.update(packet)

      #Preprocess actions for neural models
      for entID in list(actions.keys()):
         #TODO: Should this silently fail? Warning level options?
         if entID not in self.realm.players:
            continue

         ent = self.realm.players[entID]

         # Fix later -- don't allow action inputs for scripted agents
         if ent.agent.scripted:
             continue

         if not ent.alive:
            continue

         if self.config.EMULATE_FLAT_ATN:
            ent_action = {}
            idx = 0
            for atn in nmmo.Action.edges(self.config):
                ent_action[atn] = {}
                for arg in atn.edges:
                    ent_action[atn][arg] = actions[entID][idx]
                    idx += 1
            actions[entID] = ent_action

         self.actions[entID] = {}
         for atn, args in actions[entID].items():
            self.actions[entID][atn] = {}
            drop = False
            for arg, val in args.items():
               if arg.argType == nmmo.action.Fixed:
                  self.actions[entID][atn][arg] = arg.edges[val]
               elif arg == nmmo.action.Target:
                  if val >= len(ent.targets):
                      drop = True
                      continue
                  targ = ent.targets[val]
                  self.actions[entID][atn][arg] = self.realm.entity(targ)
               elif atn in (nmmo.action.Sell, nmmo.action.Use, nmmo.action.Give) and arg == nmmo.action.Item:
                  if val >= len(ent.inventory.dataframeKeys):
                      drop = True
                      continue
                  itm = [e for e in ent.inventory._item_references][val]
                  if type(itm) == Item.Gold:
                      drop = True
                      continue
                  self.actions[entID][atn][arg] = itm
               elif atn == nmmo.action.Buy and arg == nmmo.action.Item:
                  if val >= len(self.realm.exchange.dataframeKeys):
                      drop = True
                      continue
                  itm = self.realm.exchange.dataframeVals[val]
                  self.actions[entID][atn][arg] = itm
               elif __debug__: #Fix -inf in classifier and assert err on bad atns
                  assert False, f'Argument {arg} invalid for action {atn}'

            # Cull actions with bad args
            if drop and atn in self.actions[entID]:
                del self.actions[entID][atn]

      #Step: Realm, Observations, Logs
      self.dead    = self.realm.step(self.actions)
      self.actions = {}
      self.obs     = {}
      infos        = {}

      obs, rewards, dones, self.raw = {}, {}, {}, {}
      for entID, ent in self.realm.players.items():
         ob = self.realm.dataframe.get(ent)
         self.obs[entID] = ob
         if ent.agent.scripted:
            atns = ent.agent(ob)
            for atn, args in atns.items():
               for arg, val in args.items():
                  atns[atn][arg] = arg.deserialize(self.realm, ent, val)
            self.actions[entID] = atns
         else:
            obs[entID]     = ob
            rewards[entID], infos[entID] = self.reward(ent)
            dones[entID]   = False

      self.log_env()
      for entID, ent in self.dead.items():
         self.log_player(ent)

      self.realm.exchange.step()

      for entID, ent in self.dead.items():
         if ent.agent.scripted:
            continue
         rewards[ent.entID], infos[ent.entID] = self.reward(ent)

         dones[ent.entID] = False #TODO: Is this correct behavior?
         if not self.config.EMULATE_CONST_HORIZON and not self.config.RESPAWN:
            dones[ent.entID] = True

         obs[ent.entID]     = self.dummy_ob

      if self.config.EMULATE_CONST_PLAYER_N:
         emulation.pad_const_nent(self.config, self.dummy_ob, obs, rewards, dones, infos)

      if self.config.EMULATE_FLAT_OBS:
         obs = nmmo.emulation.pack_obs(obs)

      if self.config.EMULATE_CONST_HORIZON:
         assert self.realm.tick <= self.config.HORIZON
         if self.realm.tick == self.config.HORIZON:
            emulation.const_horizon(dones)

      if not len(self.realm.players.items()):
         emulation.const_horizon(dones)

      #Pettingzoo API
      self.agents = list(self.realm.players.keys())

      self.obs = obs
      return obs, rewards, dones, infos

   ############################################################################
   ### Logging
   def max(self, fn):
       return max(fn(player) for player in self.realm.players.values())

   def max_held(self, policy):
       lvls = [player.equipment.held.level.val for player in self.realm.players.values()
               if player.equipment.held is not None and player.policy == policy]

       if len(lvls) == 0:
           return 0

       return max(lvls)

   def max_item(self, policy):
       lvls = [player.equipment.item_level for player in self.realm.players.values() if player.policy == policy]

       if len(lvls) == 0:
           return 0

       return max(lvls)

   def log_env(self) -> None:
       '''Logs player data upon death
 
       This function is called automatically once per environment step
       to compute summary stats. You should not call it manually.
       Instead, override this method to customize logging.
       '''

       # This fn more or less repeats log_player once per tick
       # It was added to support eval-time logging
       # It needs to be redone to not duplicate player logging and
       # also not slow down training
       if not self.config.LOG_ENV:
           return 

       quill  = self.realm.quill

       if len(self.realm.players) == 0:
           return

       #Aggregate logs across env
       for key, fn in quill.shared.items():
          dat = defaultdict(list)
          for _, player in self.realm.players.items():
              name = player.agent.policy
              dat[name].append(fn(player))
          for policy, vals in dat.items():
              quill.log_env(f'{key}_{policy}', float(np.mean(vals)))

       if self.config.EXCHANGE_SYSTEM_ENABLED:
           for item in nmmo.systems.item.ItemID.item_ids:
               for level in range(1, 11):
                   name = item.__name__
                   key = (item, level)
                   if key in self.realm.exchange.item_listings:
                       listing = self.realm.exchange.item_listings[key]
                       quill.log_env(f'Market/{name}-{level}_Price', listing.price if listing.price else 0)
                       quill.log_env(f'Market/{name}-{level}_Volume', listing.volume if listing.volume else 0)
                       quill.log_env(f'Market/{name}-{level}_Supply', listing.supply if listing.supply else 0)
                   else:
                       quill.log_env(f'Market/{name}-{level}_Price', 0)
                       quill.log_env(f'Market/{name}-{level}_Volume', 0)
                       quill.log_env(f'Market/{name}-{level}_Supply', 0)

   def register_logs(self):
      config = self.config
      quill  = self.realm.quill

      quill.register('Basic/Lifetime', lambda player: player.history.timeAlive.val)

      if config.TASKS:
          quill.register('Task/Completed', lambda player: player.diary.completed)
          quill.register('Task/Reward' , lambda player: player.diary.cumulative_reward)
 
      else:
          quill.register('Task/Completed', lambda player: player.history.timeAlive.val)
 
      # Skills
      if config.PROGRESSION_SYSTEM_ENABLED:
         if config.COMBAT_SYSTEM_ENABLED:
             quill.register('Skill/Mage', lambda player: player.skills.mage.level.val)
             quill.register('Skill/Range', lambda player: player.skills.range.level.val)
             quill.register('Skill/Melee', lambda player: player.skills.melee.level.val)
         if config.PROFESSION_SYSTEM_ENABLED:
             quill.register('Skill/Fishing', lambda player: player.skills.fishing.level.val)
             quill.register('Skill/Herbalism', lambda player: player.skills.herbalism.level.val)
             quill.register('Skill/Prospecting', lambda player: player.skills.prospecting.level.val)
             quill.register('Skill/Carving', lambda player: player.skills.carving.level.val)
             quill.register('Skill/Alchemy', lambda player: player.skills.alchemy.level.val)
         if config.EQUIPMENT_SYSTEM_ENABLED:
             quill.register('Item/Held-Level', lambda player: player.inventory.equipment.held.level.val if player.inventory.equipment.held else 0)
             quill.register('Item/Equipment-Total', lambda player: player.equipment.total(lambda e: e.level))

      if config.EXCHANGE_SYSTEM_ENABLED:
          quill.register('Item/Wealth', lambda player: player.inventory.gold.quantity.val)

      # Item usage
      if config.PROFESSION_SYSTEM_ENABLED:
         quill.register('Item/Ration-Consumed', lambda player: player.ration_consumed)
         quill.register('Item/Poultice-Consumed', lambda player: player.poultice_consumed)
         quill.register('Item/Ration-Level', lambda player: player.ration_level_consumed)
         quill.register('Item/Poultice-Level', lambda player: player.poultice_level_consumed)

      # Market
      if config.EXCHANGE_SYSTEM_ENABLED:
         quill.register('Exchange/Player-Sells', lambda player: player.sells)
         quill.register('Exchange/Player-Buys',  lambda player: player.buys)


   def log_player(self, player) -> None:
      '''Logs player data upon death

      This function is called automatically when an agent dies
      to compute summary stats. You should not call it manually.
      Instead, override this method to customize logging.

      Args:
         player: An agent
      '''

      name = player.agent.policy
      config = self.config
      quill  = self.realm.quill
      policy = player.policy

      for key, fn in quill.shared.items():
          quill.log_player(f'{key}_{policy}', fn(player))

      # Duplicated task reward with/without name for SR calc
      if player.diary:
         if player.agent.scripted:
            player.diary.update(self.realm, player)

         quill.log_player(f'Task_Reward',     player.diary.cumulative_reward)

         for achievement in player.diary.achievements:
            quill.log_player(achievement.name, float(achievement.completed))
      else:
         quill.log_player(f'Task_Reward', player.history.timeAlive.val)

      # Used for SR
      quill.log_player('PolicyID', player.agent.policyID)
      if player.diary:
         quill.log_player(f'Task_Reward', player.diary.cumulative_reward)

   def terminal(self):
      '''Logs currently alive agents and returns all collected logs

      Automatic log calls occur only when agents die. To evaluate agent
      performance over a fixed horizon, you will need to include logs for
      agents that are still alive at the end of that horizon. This function
      performs that logging and returns the associated a data structure
      containing logs for the entire evaluation

      Args:
         ent: An agent

      Returns:
         Log datastructure
      '''

      for entID, ent in self.realm.players.entities.items():
         self.log_player(ent)

      if self.config.SAVE_REPLAY:
         self.replay.save()

      return self.realm.quill.packet

   ############################################################################
   ### Override hooks
   def reward(self, player):
      '''Computes the reward for the specified agent

      Override this method to create custom reward functions. You have full
      access to the environment state via self.realm. Our baselines do not
      modify this method; specify any changes when comparing to baselines

      Args:
         player: player object

      Returns:
         reward:
            The reward for the actions on the previous timestep of the
            entity identified by entID.
      '''
      info = {'population': player.pop}
 
      if player.entID not in self.realm.players:
         return -1, info

      if not player.diary:
         return 0, info

      achievement_rewards = player.diary.update(self.realm, player)
      reward = sum(achievement_rewards.values())

      info = {**info, **achievement_rewards}
      return reward, info
      

   ############################################################################
   ### Client data
   def render(self, mode='human') -> None:
      '''Data packet used by the renderer

      Returns:
         packet: A packet of data for the client
      '''

      assert self.has_reset, 'render before reset'
      packet = self.packet

      if not self.client:
         from nmmo.websocket import Application
         self.client = Application(self) 

      pos, cmd = self.client.update(packet)
      self.registry.step(self.obs, pos, cmd)

   def register(self, overlay) -> None:
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
               self.realm.dataframe.remove(nmmo.Serialized.Entity, entID, ent.pos)

            observations[entID] = obs
            ents[entID] = ent
            entID += 1

      return observations, ents
