from pdb import set_trace as T

from collections import defaultdict
import os

from tqdm import tqdm
import numpy as np

import gym

import torch
from torch import nn

from typing import Dict
from ray import rllib
import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.utils.spaces.flexdict import FlexDict
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.rnn_sequencing import add_time_dimension

from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.action.static import Action
from forge.blade.lib.log import InkWell

from forge.ethyr.torch import io
from forge.ethyr.torch import policy
from forge.ethyr.torch.policy import baseline

from forge.trinity import Env
from forge.trinity import evaluator 
from forge.trinity.dataframe import DataType
from forge.trinity.overlay import OverlayRegistry

import projekt

#Moved log to forge/trinity/env
class RLLibEnv(Env, rllib.MultiAgentEnv):
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

#Neural MMO observation space
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
            low=-2**16, high=2**16, shape=(nRows, nContinuous),
            dtype=DataType.CONTINUOUS)

      obs[entity.__name__]['Discrete']   = gym.spaces.Box(
            low=0, high=4096, shape=(nRows, nDiscrete),
            dtype=DataType.DISCRETE)

   obs['Entity']['N']   = gym.spaces.Box(
         low=0, high=config.N_AGENT_OBS, shape=(1,),
         dtype=DataType.DISCRETE)

   return obs

#Neural MMO action space
def actionSpace(config):
   atns = FlexDict(defaultdict(FlexDict))
   for atn in sorted(Action.edges):
      for arg in sorted(atn.edges):
         n              = arg.N(config)
         atns[atn][arg] = gym.spaces.Discrete(n)
   return atns

class RLLibEvaluator(evaluator.Base):
   '''Test-time evaluation with communication to
   the Unity3D client. Makes use of batched GPU inference'''
   def __init__(self, config, trainer):
      super().__init__(config)
      self.trainer  = trainer

      self.model    = self.trainer.get_policy('policy_0').model
      self.env      = projekt.rllib_wrapper.RLLibEnv({'config': config})

      self.env.reset(idx=0, step=False)
      self.registry = OverlayRegistry(self.env, self.model, trainer, config)
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
      self.registry.step(self.obs, pos, cmd,
            update='counts values attention wilderness'.split())

      #Step environment
      super().tick(actions)

class Policy(RecurrentNetwork, nn.Module):
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

class LogCallbacks(DefaultCallbacks):
   STEP_KEYS    = 'env_step preprocess_actions realm_step env_stim'.split()
   EPISODE_KEYS = ['env_reset']

   def init(self, episode):
      for key in LogCallbacks.STEP_KEYS + LogCallbacks.EPISODE_KEYS:
         episode.hist_data[key] = []

   def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
         policies: Dict[str, Policy],
         episode: MultiAgentEpisode, **kwargs):
      self.init(episode)

   def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
         episode: MultiAgentEpisode, **kwargs):

      env = base_env.envs[0]
      for key in LogCallbacks.STEP_KEYS:
         if not hasattr(env, key):
            continue
         episode.hist_data[key].append(getattr(env, key))

   def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
         policies: Dict[str, Policy], episode: MultiAgentEpisode, **kwargs):
      env = base_env.envs[0]
      for key in LogCallbacks.EPISODE_KEYS:
         if not hasattr(env, key):
            continue
         episode.hist_data[key].append(getattr(env, key))

      for key, val in env.terminal()['Stats'].items():
         episode.hist_data['_'+key] = val

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
      logo   = open(self.envConfig.LOGO_DIR).read().splitlines()
      epoch  = 0

      total_sample_time = 0
      total_learn_time = 0

      sep     = u'\u2595\u258f'
      block   = u'\u2591'
      top     = u'\u2581'
      bot     = u'\u2594'
      left    = u'\u258f'
      right   = u'\u2595'

      summary = left + 'Neural MMO v1.5{}Epochs: {}{}Samples: {}{}Sample Time: {:.1f}s{}Learn Time: {:.1f}s' + right
      blocks  = []

      while True:
          stats = super().train()
          self.save()

          lines = logo.copy()

          nSteps = stats['info']['num_steps_trained']

          timers             = stats['timers']
          sample_time        = timers['sample_time_ms'] / 1000
          learn_time         = timers['learn_time_ms'] / 1000
          sample_throughput  = timers['sample_throughput']
          learn_throughput   = timers['learn_throughput']

          total_sample_time += sample_time
          total_learn_time  += learn_time

          line = (left + 'Epoch: {}{}Sample: {:.1f}/s ({:.1f}s){}Train: {:.1f}/s ({:.1f}s)' + right).format(
               epoch, sep, sample_throughput, sample_time, sep, learn_throughput, learn_time)

          epoch += 1
            
          block = []
          for key, stat in stats['hist_stats'].items():
             if key.startswith('_') and len(stat) > 0:
                stat       = stat[-self.envConfig.TRAIN_BATCH_SIZE:]
                mmin, mmax = np.min(stat),  np.max(stat)
                mean, std  = np.mean(stat), np.std(stat)

                block.append(('   ' + left + '{:<12}{}Min: {:8.1f}{}Max: {:8.1f}{}Mean: {:8.1f}{}Std: {:8.1f}').format(
                      key.lstrip('_'), sep, mmin, sep, mmax, sep, mean, sep, std))

             if not self.envConfig.v: 
                continue

             if len(stat) == 0:
                continue

             lines.append('{}:: Total: {:.4f}, N: {:.4f}, Mean: {:.4f}, Std: {:.4f}'.format(
                   key, np.sum(stat), len(stat), np.mean(stat), np.std(stat)))

          if len(block) > 0:
             mmax = max(len(l) for l in block) + 1
             for idx, l in enumerate(block):
                block[idx] = ('{:<'+str(mmax)+'}').format(l + right)

             blocks.append([top*len(line), line, bot*len(line), '   ' +
                   top*(mmax-3)] + block + ['   ' + bot*(mmax-3)])
 
         
          if len(blocks) > 3:
             blocks = blocks[1:]
          
          for block in blocks:
             for line in block:
                lines.append(' ' + line)

          line = summary.format(sep, epoch, sep, nSteps, sep, total_sample_time, sep, total_learn_time)
          lines.append(' ' + top*len(line))
          lines.append(' ' + line)
          lines.append(' ' + bot*len(line))

          #Cross-platform clear screen
          os.system('cls' if os.name == 'nt' else 'clear')
          for idx, line in enumerate(lines):
             print(line)
