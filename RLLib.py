from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

import os
import gym
from gym import spaces
import time
from matplotlib import pyplot as plt
import glob

from collections import defaultdict

import ray
from ray import rllib
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune import registry
import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models import preprocessors
import argparse

from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution, TorchCategorical, TorchDiagGaussian
from ray.rllib.utils.space_utils import flatten_space
from ray.rllib.utils import try_import_tree
from ray.rllib.models import extra_spaces

from ray.rllib.policy.rnn_sequencing import add_time_dimension

import torch

from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.stimulus import node
from forge.blade.io.action.static import Action
from forge.blade.io import stimulus

from forge.blade import lib
from forge.blade import core
from forge.blade.lib.ray import init
from forge.blade.io.action import static as StaticAction
from forge.ethyr.torch import param
from forge.ethyr.torch import policy
from forge.ethyr.torch.policy import baseline
from forge.ethyr.torch import io
from forge.ethyr.torch import utils

import gym

def oneHot(i, n):
   vec = [0 for _ in range(n)]
   vec[i] = 1
   return vec

class Config(core.Config):
   #Program level args
   COMPUTE_GLOBAL_VALUES = True
   RENDER                = False

   NENT    = 256
   NPOP    = 1

   NREALM  = 256
   NWORKER = 12
   NMAPS   = 256

   #Set this high enough that you can always attack
   #Probably should sort by distance
   ENT_OBS = 15

   EMBED   = 64
   HIDDEN  = 64
   STIM    = 4
   WINDOW  = 9
   #WINDOW  = 15

class Policy(RecurrentNetwork):
   def __init__(self, *args, config=None, **kwargs):
      super().__init__(*args, **kwargs)
      self.space  = actionSpace(config).spaces
      self.h      = config.HIDDEN
      self.config = config

      #Attentional IO Networks
      self.input  = io.Input(config, policy.Input,
            baseline.Attributes, baseline.Entities)
      self.output = io.Output(config)

      #Standard recurrent hidden network and fc value network
      self.hidden = nn.LSTM(config.HIDDEN, Config.HIDDEN)
      self.valueF = nn.Linear(config.HIDDEN, 1)

   #Initial hidden state for RLlib Trainer
   def get_initial_state(self):
      return [self.valueF.weight.new(1, self.h).zero_(),
              self.valueF.weight.new(1, self.h).zero_()]

   #Initial hidden state for evalutation
   def get_initial_state_numpy(self):
      return [np.zeros(self.h, np.float32),
              np.zeros(self.h, np.float32)]

   def forward(self, input_dict, state, seq_lens):
      #Attentional input preprocessor and batching
      obs, lookup, self.attn = self.input(input_dict['obs'])
      obs   = add_time_dimension(obs, seq_lens, framework="torch")
      batch = obs.size(0)
      h, c  = state

      #Hidden Network and associated data transformations.
      #Pytorch (seq_len, batch, hidden); RLlib (batch, seq_len, hidden)
      #Optimizers batch over traj segments; Rollout workers use seq_len=1
      obs        = obs.view(batch, -1, self.h).transpose(0, 1)
      h          = h.view(batch, -1, self.h).transpose(0, 1)
      c          = c.view(batch, -1, self.h).transpose(0, 1)
      obs, state = self.hidden(obs, [h, c])
      obs        = obs.transpose(0, 1).reshape(-1, self.h)
      state      = [state[0].transpose(0, 1), state[1].transpose(0, 1)]

      #Structured attentional output postprocessor and value function
      logitDict  = self.output(obs, lookup)
      self.value = self.valueF(obs).squeeze(1)
      logits     = []

      #Flatten structured logits for RLlib
      for atnKey, atn in sorted(self.space.items()):
         for argKey, arg in sorted(atn.spaces.items()):
            logits.append(logitDict[atnKey][argKey])

      return torch.cat(logits, dim=1), state

   def value_function(self):
      return self.value

class Realm(core.Realm, rllib.MultiAgentEnv):
   def __init__(self, config):
      self.config = config['config']
          
   def reset(self):
      n   = self.config.NMAPS
      idx = np.random.randint(n)

      self.lifetimes = []
      super().__init__(self.config, idx)
      return self.step({})[0]

   '''Example environment overrides'''
   def step(self, decisions, values={}, attns={}):
      #Postprocess actions
      actions = {}
      for entID in list(decisions.keys()):
         actions[entID] = defaultdict(dict)
         if entID in self.dead:
            continue

         ents = self.raw[entID][('Entity',)]
         for atn, args in decisions[entID].items():
            for arg, val in args.items():
               val = int(val)
               if len(arg.edges) > 0:
                  actions[entID][atn][arg] = arg.edges[val]
               elif val < len(ents):
                  #Note: you are not masking over padded agents yet.
                  #This will make learning attacks very hard
                  actions[entID][atn][arg] = ents[val]
               else:
                  actions[entID][atn][arg] = ents[0]

      obs, rewards, dones, _ = super().step(actions, values, attns)

      for ent in self.dead:
         self.lifetimes.append(ent.history.timeAlive.val)
         if not self.config.RENDER and len(self.lifetimes) >= 1000:
            lifetime = np.mean(self.lifetimes)
            print('Lifetime: {}'.format(lifetime))
            dones['__all__'] = True

      return obs, rewards, dones, {}

class SanePPOTrainer(ppo.PPOTrainer):
   def __init__(self, env, path, config):
      super().__init__(env=env, config=config)
      self.saveDir = path

   def save(self):
      savedir = super().save(self.saveDir)
      with open('experiment/path.txt', 'w') as f:
         f.write(savedir)
      print('Saved to: {}'.format(savedir))
      return savedir

   def restore(self):
      with open('experiment/path.txt') as f:
         path = f.read()
      print('Loading from: {}'.format(path))
      super().restore(path)

   def policyID(self, idx):
      return 'policy_{}'.format(idx)

   def model(self, policyID):
      return self.get_policy(policyID).model

   def defaultModel(self):
      return self.model(self.policyID(0))

   def train(self):
      epoch = 0
      while True:
          stats = super().train()
          self.save()

          nSteps = stats['info']['num_steps_trained']
          print('Epoch: {}, Samples: {}'.format(epoch, nSteps))
          epoch += 1

class Evaluator:
   def __init__(self, trainer, env, config):
      self.obs  = env.reset()
      self.env  = env
      self.done = {}

      self.config   = config
      config.RENDER = True

      self.trainer  = trainer
      self.values()

   #Start a persistent Twisted environment server
   def run(self):
      from forge.embyr.twistedserver import Application
      Application(self.env, self.tick)

   #Compute actions and overlays for a single timestep
   def tick(self):
      atns, values, attns = {}, {}, defaultdict(dict)
      for agentID, ob in self.obs.items():
         if agentID in self.done and self.done[agentID]:
            continue

         ent = self.env.desciples[agentID]
         atns[agentID], model = self.action(agentID, ent, ob)
         values[agentID] = float(model.value)

         tiles  = self.env.raw[agentID][('Tile',)]
         for tile, a in zip(tiles, model.attn):
            attns[ent][tile] = float(a)

      self.obs, rewards, self.done, _ = self.env.step(atns, values, attns)
 
   #Compute actions for a single agent
   def action(self, agentID, ent, obs, mock=False):
      idx      = 0 if mock else agentID % self.config.NPOP
      policyID = self.trainer.policyID(idx)

      model = self.trainer.model(policyID)
 
      init  = mock or not hasattr(ent, 'state')
      if init:
         state = model.get_initial_state_numpy()
      else:
         state = ent.state

      #RLlib bug here -- you have to modify their Policy line
      #169 to not return action instead of [action]
      atns, state, _ = trainer.compute_action(
            obs, policy_id=policyID, state=state)

      if not mock:
         ent.state = state

      return atns, model
  
   #Compute a global value function map. This requires ~6400 forward
   #passes and a ton of environment deep copy operations, which will 
   #take several minutes. You can disable this computation in the config
   def values(self):
      values = np.zeros(self.env.size)
      if not self.config.COMPUTE_GLOBAL_VALUES:
         self.env.setGlobalValues(values)
         return

      print('Computing value map...')
      values = np.zeros(self.env.size)
      for obs, stim in self.env.getValStim():
         env, ent   = stim
         r, c       = ent.base.pos

         atn, model   = self.action(ent.entID, ent, obs, mock=True)
         values[r, c] = float(model.value)
 
      self.env.setGlobalValues(values)
      print('Value map computed')

#Neural MMO observation space
def observationSpace(config):
   obs = extra_spaces.FlexDict({})
   for entity, attrList in sorted(Stimulus):
      attrDict = extra_spaces.FlexDict({})
      for attr, val in sorted(attrList):
         attrDict[attr] = val(config).space
      n = attrList.N(config)
      obs[entity] = extra_spaces.Repeated(attrDict, max_len=n)
   return obs

#Neural MMO action space
def actionSpace(config):
   atns = extra_spaces.FlexDict(defaultdict(extra_spaces.FlexDict))
   for atn in sorted(Action.edges):
      for arg in sorted(atn.edges):
         n = arg.N(config)
         atns[atn][arg] = gym.spaces.Discrete(n)
   return atns

#Generate RLlib policy
def gen_policy(config, i):
   obs  = observationSpace(config)
   atns = actionSpace(config)

   params = {
               "agent_id": i,
               "obs_space_dict": obs,
               "act_space_dict": atns
            }
 
   return (None, obs, atns, params)

#Instantiate a new environment
def env_creator(config):
   return Realm(config)

if __name__ == '__main__':
   #Setup and config
   torch.set_num_threads(1)
   config = Config()
   ray.init()

   #RLlib registry
   ModelCatalog.register_custom_model('test_model', Policy)
   registry.register_env("custom", env_creator)

   #Create policies
   policyMap = lambda i: 'policy_{}'.format(i % config.NPOP)
   policies  = {policyMap(i): gen_policy(config, i) for i in range(config.NPOP)}

   #Instantiate monolithic RLlib Trainer object.
   trainer = SanePPOTrainer(env="custom", path='experiment', config={
      'num_workers': 5,
      'num_gpus': 1,
      'num_envs_per_worker': 1,
      'train_batch_size': 5000,
      'rollout_fragment_length': 100,
      'sgd_minibatch_size': 128,
      'num_sgd_iter': 1,
      'use_pytorch': True,
      'horizon': np.inf,
      'soft_horizon': False, 
      'no_done_at_end': False,
      'env_config': {
         'config': config
      },
      'multiagent': {
         "policies": policies,
         "policy_mapping_fn": policyMap
      },
      'model': {
         'custom_model': 'test_model',
         'custom_options': {'config': config}
      },
   })

   #Print model size
   utils.modelSize(trainer.defaultModel())

   trainer.restore()
   #trainer.train()
   Evaluator(trainer, env_creator({'config': config}), config).run()
