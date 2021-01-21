from pdb import set_trace as T

from collections import defaultdict
from itertools import chain
import os

from tqdm import tqdm
import numpy as np

import gym

import torch
from torch import nn

from ray import rllib
import ray.rllib.agents.ppo.ppo as ppo
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

   def step(self, decisions, omitDead=False, preprocessActions=True):
      obs, rewards, dones, infos = super().step(decisions,
            omitDead=omitDead, preprocessActions=preprocessActions)

      t, mmean = len(self.lifetimes), np.mean(self.lifetimes)
      if not self.config.EVALUATE and t >= self.config.TRAIN_HORIZON:
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

################################################################################
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

      self.env.reset(idx=0, step=False)
      self.registry = RLlibOverlayRegistry(
            config, self.env).init(trainer, self.model)
      self.obs      = self.env.step({})[0]
      self.state    = {} 

   def tick(self, pos, cmd):
      '''Compute actions and overlays for a single timestep
      Args:
          pos: Camera position (r, c) from the server)
          cmd: Console command from the server
      '''
      #Compute batch of actions
      actions, self.state, _ = self.trainer.compute_actions(
            self.obs, state=self.state, policy_id='policy_0')
      self.registry.step(self.obs, pos, cmd)

      #Step environment
      super().tick(actions)

class SanePPOTrainer(ppo.PPOTrainer):
   '''Small utility class on top of RLlib's base trainer'''
   def __init__(self, env, path, config):
      super().__init__(env=env, config=config)
      self.envConfig = config['env_config']['config']
      self.saveDir   = path

   def save(self):
      '''Save model to file. Note: RLlib does not let us chose save paths'''
      savedir = super().save(self.saveDir)
      with open('experiment/path.txt', 'w') as f:
         f.write(savedir)
      print('Saved to: {}'.format(savedir))
      return savedir

   def restore(self, model):
      '''Restore model from path'''
      if model is None:
         print('Training from scratch...')
         return
      if model == 'current':
         with open('experiment/path.txt') as f:
            path = f.read().splitlines()[0]
      else:
         path = 'experiment/{}/checkpoint'.format(model)

      print('Loading from: {}'.format(path))
      super().restore(path)

   def policyID(self, idx):
      return 'policy_{}'.format(idx)

   def model(self, policyID):
      return self.get_policy(policyID).model

   def defaultModel(self):
      return self.model(self.policyID(0))

   def train(self):
      '''Train forever, printing per epoch'''
      config = self.envConfig

      logo   = open(config.PATH_LOGO).read().splitlines()

      total_sample_time = 0
      total_learn_time = 0

      epoch   = 0
      blocks  = []

      while True:
          #Train model
          stats = super().train()
          self.save()
          epoch += 1

          if epoch == 1:
            continue

          #Time Stats
          timers             = stats['timers']
          sample_time        = timers['sample_time_ms'] / 1000
          learn_time         = timers['learn_time_ms'] / 1000
          sample_throughput  = timers['sample_throughput']
          learn_throughput   = timers['learn_throughput']

          #Summary
          nSteps             = stats['info']['num_steps_trained']
          total_sample_time += sample_time
          total_learn_time  += learn_time
          summary = formatting.box([formatting.line(
                title  = 'Neural MMO v1.5',
                keys   = ['Epochs', 'kSamples', 'Sample Time', 'Learn Time'],
                vals   = [epoch, nSteps/1000, total_sample_time, total_learn_time],
                valFmt = '{:.1f}')])

          #Block Title
          sample_stat = '{:.1f}/s ({:.1f}s)'.format(sample_throughput, sample_time)
          learn_stat  = '{:.1f}/s ({:.1f}s)'.format(learn_throughput, learn_time)
          header = formatting.box([formatting.line(
                keys   = 'Epoch Sample Train'.split(),
                vals   = [epoch, sample_stat, learn_stat],
                valFmt = '{}')])

          #Format stats
          hist    = stats['hist_stats']
          timings = {k: v for k, v in hist.items() if not k.startswith('_')}
          stats   = {k.lstrip('_'): v[-config.TRAIN_BATCH_SIZE:] for k, v in 
               stats['hist_stats'].items() if k.startswith('_')}

          lines = formatting.stats(stats)
          if config.v:
             lines += formatting.timings(timings)

          #Extend blocks
          if len(lines) > 0:
             blocks.append(header + formatting.box(lines, indent=4))
             
          if len(blocks) > 3:
             blocks = blocks[1:]
          
          #Assemble Summary Bar Title
          lines = logo.copy() + list(chain.from_iterable(blocks)) + summary

          #Cross-platform clear screen
          os.system('cls' if os.name == 'nt' else 'clear')
          for idx, line in enumerate(lines):
             print(line)

################################################################################
### RLlib Overlays
class RLlibOverlay(Overlay):
   def __init__(self, config, realm, trainer, model):
      super().__init__(config, realm)
      self.trainer = trainer
      self.model   = model

class RLlibOverlayRegistry(OverlayRegistry):
   def __init__(self, config, realm):
      super().__init__(config, realm)

      self.overlays['values']       = Values
      self.overlays['globalValues'] = GlobalValues
      self.overlays['attention']    = Attention

class Attention(RLlibOverlay):
   def register(self, obs):
      '''Computes local attentional maps with respect to each agent'''
      attentions = defaultdict(list)
      for idx, agentID in enumerate(obs):
         ent   = self.realm.realm.players[agentID]
         rad   = self.config.STIM
         r, c  = ent.pos

         tiles = self.realm.realm.map.tiles[r-rad:r+rad+1, c-rad:c+rad+1].ravel()
         for tile, a in zip(tiles, self.model.attention()[idx]):
            attentions[tile].append(float(a))

      data = np.zeros((self.R, self.C))
      tiles = self.realm.realm.map.tiles
      for r, tList in enumerate(tiles):
         for c, tile in enumerate(tList):
            if tile not in attentions:
               continue
            data[r, c] = np.mean(attentions[tile])

      colorized = overlay.twoTone(data)
      self.realm.registerOverlay(colorized)

class Values(RLlibOverlay):
   def update(self, obs):
      '''Computes a local value function by painting tiles as agents
      walk over them. This is fast and does not require additional
      network forward passes'''
      for idx, agentID in enumerate(obs):
         r, c = self.realm.realm.players[agentID].base.pos
         self.values[r, c] = float(self.model.value_function()[idx])

   def register(self, obs):
      colorized = overlay.twoTone(self.values[:, :])
      self.realm.registerOverlay(colorized)

class GlobalValues(RLlibOverlay):
   def init(self):
      '''Compute a global value function map. This requires ~6400 forward
      passes and may take up to a minute. You can disable this computation
      in the config file'''
      if self.trainer is None:
         return

      print('Computing value map...')
      values    = np.zeros(self.realm.size)
      model     = self.trainer.get_policy('policy_0').model
      obs, ents = self.realm.getValStim()

      #Compute actions to populate model value function
      BATCH_SIZE = 128
      batch = {}
      final = list(obs.keys())[-1]
      for agentID in tqdm(obs):
         batch[agentID] = obs[agentID]
         if len(batch) == BATCH_SIZE or agentID == final:
            self.trainer.compute_actions(batch, state={}, policy_id='policy_0')
            for idx, agentID in enumerate(batch):
               r, c = ents[agentID].base.pos
               values[r, c] = float(self.model.value_function()[idx])
            batch = {}

      print('Value map computed')
      self.colorized = overlay.twoTone(values)

   def register(self, obs):
      if not hasattr(self, 'colorized'):
         print('Initializing Global Values. This requires one NN pass per tile')
         self.init()

      self.realm.registerOverlay(self.colorized)

################################################################################
### Performance and Debug Logging
class RLlibLogCallbacks(DefaultCallbacks):
   STEP_KEYS    = 'env_step preprocess_actions realm_step env_stim'.split()
   EPISODE_KEYS = ['env_reset']

   def init(self, episode):
      for key in RLlibLogCallbacks.STEP_KEYS + RLlibLogCallbacks.EPISODE_KEYS:
         episode.hist_data[key] = []

   def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
      self.init(episode)

   def on_episode_step(self, *, worker, base_env, episode, **kwargs):
      env = base_env.envs[0]
      for key in RLlibLogCallbacks.STEP_KEYS:
         if not hasattr(env, key):
            continue
         episode.hist_data[key].append(getattr(env, key))

   def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
      env = base_env.envs[0]
      for key in RLlibLogCallbacks.EPISODE_KEYS:
         if not hasattr(env, key):
            continue
         episode.hist_data[key].append(getattr(env, key))

      for key, val in env.terminal()['Stats'].items():
         episode.hist_data['_'+key] = val
