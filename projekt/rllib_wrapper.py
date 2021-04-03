from pdb import set_trace as T

from collections import defaultdict
from itertools import chain
import shutil
import contextlib
import time
import os
import re

from tqdm import tqdm
import numpy as np

import gym

import torch
from torch import nn

from ray import rllib
import ray.rllib.agents.ppo.ppo as ppo
import ray.rllib.agents.ppo.appo as appo
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.utils.spaces.flexdict import FlexDict
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.action.static import Action
from forge.blade.lib import overlay

from forge.ethyr.torch.policy import baseline

from forge.trinity import Env, evaluator, formatting
from forge.trinity.dataframe import DataType
from forge.trinity.overlay import Overlay, OverlayRegistry

###############################################################################
### RLlib Env Wrapper
class RLlibEnv(Env, rllib.MultiAgentEnv):
   def __init__(self, config):
      self.config = config['config']
      super().__init__(self.config)

   def reward(self, ent):
      reward, pop = 0, ent.pop
      for p in self.dead.values():
         if p.population == pop:
            reward -= 1

      return reward

      #if entID not in self.realm.players:
      #   return -1
      #return 0

      #player = self.realm.players[entID]
      #return player.achievements.update(self.realm, player) / 15

   def step(self, decisions, omitDead=False, preprocessActions=True):
      obs, rewards, dones, infos = super().step(
            decisions, omitDead, preprocessActions)

      config = self.config
      dones['__all__'] = False
      test = config.EVALUATE or config.RENDER
      
      horizon    = self.realm.tick >= config.TRAIN_HORIZON
      population = len(self.realm.players) == 0
      if not test and (horizon or population):
         dones['__all__'] = True

      return obs, rewards, dones, infos

def observationSpace(config):
   obs = FlexDict(defaultdict(FlexDict))
   for entity in sorted(Stimulus.values()):
      nRows       = entity.N(config)
      nContinuous = 0
      nDiscrete   = 0

      for _, attr in entity:
         if attr.DISCRETE:
            nDiscrete += 1
         if attr.CONTINUOUS:
            nContinuous += 1

      obs[entity.__name__]['Continuous'] = gym.spaces.Box(
            low=-2**20, high=2**20, shape=(nRows, nContinuous),
            dtype=DataType.CONTINUOUS)

      obs[entity.__name__]['Discrete']   = gym.spaces.Box(
            low=0, high=4096, shape=(nRows, nDiscrete),
            dtype=DataType.DISCRETE)

   obs['Entity']['N']   = gym.spaces.Box(
         low=0, high=config.N_AGENT_OBS, shape=(1,),
         dtype=DataType.DISCRETE)

   return obs

def actionSpace(config):
   atns = FlexDict(defaultdict(FlexDict))
   for atn in sorted(Action.edges):
      for arg in sorted(atn.edges):
         n              = arg.N(config)
         atns[atn][arg] = gym.spaces.Discrete(n)
   return atns

###############################################################################
### RLlib Policy, Evaluator, and Trainer wrappers
class RLlibPolicy(RecurrentNetwork, nn.Module):
   '''Wrapper class for using our baseline models with RLlib'''
   def __init__(self, *args, **kwargs):
      self.config = kwargs.pop('config')
      super().__init__(*args, **kwargs)
      nn.Module.__init__(self)

      self.space  = actionSpace(self.config).spaces

      #Select appropriate baseline model
      if self.config.MODEL == 'attentional':
         self.model  = baseline.Attentional(self.config)
      elif self.config.MODEL == 'convolutional':
         self.model  = baseline.Simple(self.config)
      else:
         self.model  = baseline.Recurrent(self.config)

   #Initial hidden state for RLlib Trainer
   def get_initial_state(self):
      return [self.model.valueF.weight.new(1, self.config.HIDDEN).zero_(),
              self.model.valueF.weight.new(1, self.config.HIDDEN).zero_()]

   def forward(self, input_dict, state, seq_lens):
      logitDict, state = self.model(input_dict['obs'], state, seq_lens)

      logits = []
      #Flatten structured logits for RLlib
      for atnKey, atn in sorted(self.space.items()):
         for argKey, arg in sorted(atn.spaces.items()):
            logits.append(logitDict[atnKey][argKey])

      return torch.cat(logits, dim=1), state

   def value_function(self):
      return self.model.value

   def attention(self):
      return self.model.attn

class RLlibEvaluator(evaluator.Base):
   '''Test-time evaluation with communication to
   the Unity3D client. Makes use of batched GPU inference'''
   def __init__(self, config, trainer):
      super().__init__(config)
      self.trainer  = trainer

      self.model    = self.trainer.get_policy('policy_0').model
      self.env      = RLlibEnv({'config': config})
      self.state    = {} 

   def render(self):
      self.obs = self.env.reset(idx=1)
      self.registry = RLlibOverlayRegistry(
            self.config, self.env).init(self.trainer, self.model)
      super().render()

   def tick(self, pos, cmd):
      '''Simulate a single timestep

      Args:
          pos: Camera position (r, c) from the server)
          cmd: Console command from the server
      '''
      if len(self.obs) == 0:
         actions = {}
      else:
         actions, self.state, _ = self.trainer.compute_actions(
             self.obs, state=self.state, policy_id='policy_0')

      super().tick(self.obs, actions, pos, cmd)

class SanePPOTrainer(ppo.PPOTrainer):
   '''Small utility class on top of RLlib's base trainer'''
   def __init__(self, config):
      self.envConfig = config['env_config']['config']
      super().__init__(env=self.envConfig.ENV_NAME, config=config)
      self.training_logs = {}

   def save(self):
      '''Save model to file. Note: RLlib does not let us chose save paths'''
      config   = self.envConfig
      saveFile = super().save(config.PATH_CHECKPOINTS)
      saveDir  = os.path.dirname(saveFile)
      
      #Clear current save dir
      shutil.rmtree(config.PATH_MODEL, ignore_errors=True)
      os.mkdir(config.PATH_MODEL)

      #Copy checkpoints
      for f in os.listdir(saveDir):
         stripped = re.sub('-\d+', '', f)
         src      = os.path.join(saveDir, f)
         dst      = os.path.join(config.PATH_MODEL, stripped) 
         shutil.copy(src, dst)

      print('Saved to: {}'.format(saveDir))

   def restore(self):
      '''Restore model from path'''
      self.training_logs = np.load(
            self.envConfig.PATH_TRAINING_DATA,
            allow_pickle=True).item()

      path = os.path.join(
            self.envConfig.PATH_MODEL,
            'checkpoint')

      print('Loading model from: {}'.format(path))
      super().restore(path)

   def policyID(self, idx):
      return 'policy_{}'.format(idx)

   def model(self, policyID):
      return self.get_policy(policyID).model

   def defaultModel(self):
      return self.model(self.policyID(0))

   def train(self):
      '''Train forever, printing per epoch'''
      training_logs = self.training_logs
      config        = self.envConfig

      logo          = open(config.PATH_LOGO).read().splitlines()

      model         = config.MODEL if config.MODEL is not None else 'current'
      trainPath     = config.PATH_TRAINING_DATA

      total_sample_time = 0
      total_learn_time  = 0
      total_steps       = 0
      total_time        = 0
      start_time        = time.time()

      blocks  = []

      for epoch in range(config.TRAIN_EPOCHS):
          #Train model
          stats = super().train()
          self.save()

          #Compute stats
          info               = stats['info']
          timers             = stats['timers']

          steps              = info['num_agent_steps_trained'] - total_steps
          total_steps        = info['num_agent_steps_trained']

          sample_time        = timers['sample_time_ms'] / 1000
          learn_time         = timers['learn_time_ms'] / 1000

          sample_throughput  = steps / sample_time
          learn_throughput   = steps / learn_time
         
          total_sample_time += sample_time
          total_learn_time  += learn_time
          total_time         = time.time() - start_time

          #Summary
          summary = formatting.box([formatting.line(
                title  = 'Neural MMO v1.5',
                keys   = ['Epochs', 'kSamples', 'Sample Time', 'Learn Time'],
                vals   = [epoch, total_steps/1000, total_sample_time, total_learn_time],
                valFmt = '{:.1f}')])

          #Block Title
          sample_stat = '{:.1f}/s ({:.1f}s)'.format(sample_throughput, sample_time)
          learn_stat  = '{:.1f}/s ({:.1f}s)'.format(learn_throughput, learn_time)
          header = formatting.box([formatting.line(
                keys   = 'Epoch Sample Train'.split(),
                vals   = [epoch, sample_stat, learn_stat],
                valFmt = '{}')])

          #Format stats (RLlib callback format limitation)
          for k, vals in stats['hist_stats'].items():
             if not k.startswith('_'):
                continue
             k                 = k.lstrip('_')
             track, stat       = re.split('_', k)

             if track not in training_logs:
                training_logs[track] = {}

             if stat not in training_logs[track]:
                training_logs[track][stat] = []

             training_logs[track][stat] += vals

          np.save(trainPath, {
               'logs': training_logs,
               'sample_time': total_sample_time,
               'learn_time': total_learn_time})

          #Representation for CLI
          cli = {}
          for track, stats in training_logs.items():
             cli[track] = {}
             for stat, vals in stats.items():
                mmean = np.mean(vals[-config.TRAIN_SUMMARY_ENVS:])
                cli[track][stat] = mmean

          lines = formatting.precomputed_stats(cli)
          if config.v:
             lines += formatting.timings(timings)

          #Extend blocks
          if len(lines) > 0:
             lines = formatting.box(lines, indent=4) 
             blocks.append(header + lines)
          else:
             blocks.append(header)
             
          if len(blocks) > 3:
             blocks = blocks[1:]
          
          #Assemble Summary Bar Title
          lines = logo.copy() + list(chain.from_iterable(blocks)) + summary

          #Cross-platform clear screen
          os.system('cls' if os.name == 'nt' else 'clear')
          for idx, line in enumerate(lines):
             print(line)

###############################################################################
### RLlib Overlays
class RLlibOverlayRegistry(OverlayRegistry):
   '''Host class for RLlib Map overlays'''
   def __init__(self, config, realm):
      super().__init__(config, realm)

      self.overlays['values']       = Values
      self.overlays['attention']    = Attention
      self.overlays['tileValues']   = TileValues
      self.overlays['entityValues'] = EntityValues

class RLlibOverlay(Overlay):
   '''RLlib Map overlay wrapper'''
   def __init__(self, config, realm, trainer, model):
      super().__init__(config, realm)
      self.trainer = trainer
      self.model   = model

class Attention(RLlibOverlay):
   def register(self, obs):
      '''Computes local attentional maps with respect to each agent'''
      tiles      = self.realm.realm.map.tiles
      players    = self.realm.realm.players

      attentions = defaultdict(list)
      for idx, playerID in enumerate(obs):
         if playerID not in players:
            continue
         player = players[playerID]
         r, c   = player.pos

         rad     = self.config.STIM
         obTiles = self.realm.realm.map.tiles[r-rad:r+rad+1, c-rad:c+rad+1].ravel()

         for tile, a in zip(obTiles, self.model.attention()[idx]):
            attentions[tile].append(float(a))

      sz    = self.config.TERRAIN_SIZE
      data  = np.zeros((sz, sz))
      for r, tList in enumerate(tiles):
         for c, tile in enumerate(tList):
            if tile not in attentions:
               continue
            data[r, c] = np.mean(attentions[tile])

      colorized = overlay.twoTone(data)
      self.realm.register(colorized)

class Values(RLlibOverlay):
   def update(self, obs):
      '''Computes a local value function by painting tiles as agents
      walk over them. This is fast and does not require additional
      network forward passes'''
      players = self.realm.realm.players
      for idx, playerID in enumerate(obs):
         if playerID not in players:
            continue
         r, c = players[playerID].base.pos
         self.values[r, c] = float(self.model.value_function()[idx])

   def register(self, obs):
      colorized = overlay.twoTone(self.values[:, :])
      self.realm.register(colorized)

def zeroOb(ob, key):
   for k in ob[key]:
      ob[key][k] *= 0

class GlobalValues(RLlibOverlay):
   '''Abstract base for global value functions'''
   def init(self, zeroKey):
      if self.trainer is None:
         return

      print('Computing value map...')
      model     = self.trainer.get_policy('policy_0').model
      obs, ents = self.realm.dense()
      values    = 0 * self.values

      #Compute actions to populate model value function
      BATCH_SIZE = 128
      batch = {}
      final = list(obs.keys())[-1]
      for agentID in tqdm(obs):
         ob             = obs[agentID]
         batch[agentID] = ob
         zeroOb(ob, zeroKey)
         if len(batch) == BATCH_SIZE or agentID == final:
            self.trainer.compute_actions(batch, state={}, policy_id='policy_0')
            for idx, agentID in enumerate(batch):
               r, c         = ents[agentID].base.pos
               values[r, c] = float(self.model.value_function()[idx])
            batch = {}

      print('Value map computed')
      self.colorized = overlay.twoTone(values)

   def register(self, obs):
      print('Computing Global Values. This requires one NN pass per tile')
      self.init()

      self.realm.register(self.colorized)

class TileValues(GlobalValues):
   def init(self, zeroKey='Entity'):
      '''Compute a global value function map excluding other agents. This
      requires a forward pass for every tile and will be slow on large maps'''
      super().init(zeroKey)

class EntityValues(GlobalValues):
   def init(self, zeroKey='Tile'):
      '''Compute a global value function map excluding tiles. This
      requires a forward pass for every tile and will be slow on large maps'''
      super().init(zeroKey)


###############################################################################
### Logging
class RLlibLogCallbacks(DefaultCallbacks):
   def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
      assert len(base_env.envs) == 1, 'One env per worker'
      env    = base_env.envs[0]
      config = env.config

      for key, vals in env.terminal()['Stats'].items():
         logs = episode.hist_data
         key  = '_' + key

         logs[key + '_Min']  = [np.min(vals)]
         logs[key + '_Max']  = [np.max(vals)]
         logs[key + '_Mean'] = [np.mean(vals)]
         logs[key + '_Std']  = [np.std(vals)]
