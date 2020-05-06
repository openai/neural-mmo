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
from ray.rllib.models import preprocessors
import argparse

from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution, TorchCategorical, TorchDiagGaussian
from ray.rllib.utils.space_utils import flatten_space
from ray.rllib.utils import try_import_tree

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
#from forge.ethyr.torch.policy.baseline import IO
from forge.ethyr.torch import policy
from forge.ethyr.torch.policy import baseline
from forge.ethyr.torch import io
from forge.ethyr.torch import utils

def oneHot(i, n):
   vec = [0 for _ in range(n)]
   vec[i] = 1
   return vec

class Config(core.Config):
   SIMPLE  = True
   NENT    = 256
   NPOP    = 8

   NREALM  = 256
   NWORKER = 12

   EMBED   = 32
   HIDDEN  = 64
   STIM    = 4
   WINDOW  = 9
   #WINDOW  = 15

   #Set this high enough that you can always attack
   #Probably should sort by distance
   ENT_OBS = 10

   OPTIM_STEPS = 128
   DEVICE      = 'cpu'

class Policy(TorchModelV2, nn.Module):
   def __init__(self, *args, **kwargs):
      TorchModelV2.__init__(self, *args, **kwargs)
      nn.Module.__init__(self)

      if Config.SIMPLE:
         self.fc     = nn.Linear(564, 4)
         self.valueF = nn.Linear(564, 1)
      else:
         self.input = io.Input(Config, policy.TaggedInput,
               baseline.Attributes, baseline.Entities)
         self.output = io.Output(Config)
         self.valueF = nn.Linear(Config.HIDDEN, 1)

   def forward(self, input_dict, state, seq_lens):
      obs           = input_dict['obs']

      if Config.SIMPLE:
         obs           = input_dict['obs_flat']
         self.value    = self.valueF(obs).squeeze(1)
         return self.fc(obs), []
 
      state, lookup = self.input(obs)
      logits        = self.output(state, lookup)

      self.value    = self.valueF(state).squeeze(1)

      flatLogits = []
      inputLens = []
      childDistributions=[]
      space = actionSpace()
      for atnKey, atn in space.spaces.items():
         for argKey, arg in atn.spaces.items():
            arg = logits[atnKey][argKey]
            flatLogits.append(arg)
            inputLens.append(arg.size(-1))
            childDistributions.append(TorchCategorical)

      flatLogits = torch.cat(flatLogits, dim=1)

      distrib = TorchMultiActionDistribution(
         flatLogits,
         model=self,
         child_distributions=childDistributions,
         input_lens=inputLens,
         action_space=actionSpace()) 

      atns = distrib.sample()
      tree = try_import_tree()
      flat = tree.flatten(atns)
      return flatLogits, []

   def value_function(self):
      return self.value

class Realm(core.Realm, rllib.MultiAgentEnv):
   def __init__(self, config, idx=0):
      super().__init__(config, idx)
      self.lifetimes = []
           
   def reset(self):
      #assert self.tick == 0
      return self.step({})[0]

   '''Example environment overrides'''
   def step(self, decisions):
      #Postprocess actions
      '''
      if len(decisions) > 0:
         print(   
               decisions.keys(),
               self.desciples.keys(),
               [e.entID for e in self.dead])
      '''
      for entID in list(decisions.keys()):
         if entID in self.dead:
            continue

         atn = decisions[entID]
         if Config.SIMPLE:
            direction = atn
         else:
            style     = int(atn['Attack']['Style'])
            target    = int(atn['Attack']['Target'])
            direction = int(atn['Move']['Direction'])

         atn = {}
         atn[StaticAction.Move]   = {
               StaticAction.Direction:
                     StaticAction.Direction.edges[direction]
            }

         #Note: you are not masking over padded agents yet.
         #This will make learning attacks very hard
         if not Config.SIMPLE:
            ents = self.raw[entID][('Entity',)]
            if target < len(ents):
               atn[StaticAction.Attack] = {
                     StaticAction.Style:
                           StaticAction.Style.edges[style],
                     StaticAction.Target:
                           ents[target]
                  }
            
         decisions[entID] = atn
      
      obs, rewards, dones, _ = super().step(decisions)

      for ent in self.dead:
         self.lifetimes.append(ent.history.timeAlive.val)
         if len(self.lifetimes) >= 1000:
            lifetime = np.mean(self.lifetimes)
            self.lifetimes = []
            print('Lifetime: {}'.format(lifetime))

      return obs, rewards, dones, {}

def env_creator(args):
   return Realm(Config(), idx=0)

def actionSpace():
   atns = defaultdict(dict)
   for atn in Action.edges:
      for arg in atn.edges:
         n = len(arg.edges)
         #Quick hack for testing
         if arg.__name__ == 'Target': 
            atns[atn.__name__][arg.__name__] = gym.spaces.Discrete(20)
         else:
            atns[atn.__name__][arg.__name__] = gym.spaces.Discrete(n)
      atns[atn.__name__] = spaces.Dict(atns[atn.__name__])
   atns = spaces.Dict(atns)
   return atns

def gen_policy(env, i):
   obs, entityDict  = {}, {}
   for entity, attrList in Stimulus:
      attrDict = {}
      for attr, val in attrList:
         if issubclass(val, node.Discrete):
            attrDict[attr] = spaces.Box(
                  low=0, high=val(Config()).range, shape=(1,))
         elif issubclass(val, node.Continuous):
            attrDict[attr] = spaces.Box(
                  low=-1, high=1, shape=(1,))
      entityDict[entity] = spaces.Dict(attrDict)

   key  = tuple(['Tile'])
   obs[key] = spaces.Tuple([entityDict[key] for _ in range(Config.WINDOW**2)])

   key  = tuple(['Entity'])
   obs[key] = spaces.Tuple([entityDict[key] for _ in range(20)])

   obs    = spaces.Dict(obs)
   if Config.SIMPLE:
      atns   = spaces.Discrete(4)
   else:
      atns   = actionSpace()
   params = {
               "agent_id": i,
               "obs_space_dict": obs,
               "act_space_dict": atns
            }
 
   return (None, obs, atns, params)

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

def train(trainer):
   epoch = 0
   while True:
       stats = trainer.train()
       trainer.save()

       nSteps = stats['info']['num_steps_trained']
       print('Epoch: {}, Samples: {}'.format(epoch, nSteps))
       epoch += 1

class Evaluator:
   def __init__(self, env, trainer):
      self.trainer = trainer

      self.env     = env
      self.obs     = env.reset()
      self.done    = {}

   def run(self):
      from forge.embyr.twistedserver import Application
      Application(self.env, self.tick)
      self.tick()

   def tick(self):
      atns = {}
      for agentID, ob in self.obs.items():
         if agentID in self.done and self.done[agentID]:
            continue

         atn = trainer.compute_action(self.obs[agentID],
               policy_id='policy_{}'.format(0))
         atns[agentID] = atn
         
      self.obs, rewards, self.done, _ = self.env.step(atns)
 
if __name__ == '__main__':
   #lib.ray.init(Config, 'local')
   lib.ray.init(Config, 'default') 
   #ray.init(local_mode=True)

   ModelCatalog.register_custom_model('test_model', Policy)
   registry.register_env("custom", env_creator)
   env      = env_creator({})

   policies = {"policy_{}".format(i): gen_policy(env, i) for i in range(1)}
   keys     = list(policies.keys())

   '''
   Epoch: 50, Samples: 204000
   (pid=12436) Lifetime: 27.006
   (pid=12430) Lifetime: 26.223
   (pid=12435) Lifetime: 26.427
   (pid=12433) Lifetime: 26.382
   '''


   #Note: you are on rllib 0.8.2. 0.8.4 seems to break some stuff
   #Note: sample_batch_size and rollout_fragment_length are overriden
   #by 'complete_episodes'
   #'batch_mode': 'complete_episodes',
   #Do not need 'no_done_at_end': True because horizon is inf
   trainer = SanePPOTrainer(env="custom", path='experiment', config={
      'num_workers': 4,
      'rollout_fragment_length': 100,
      'train_batch_size': 4000,
      'sgd_minibatch_size': 128,
      'num_sgd_iter': 1,
      'use_pytorch': True,
      'horizon': np.inf,
      'soft_horizon': True, 
      "multiagent": {
         "policies": policies,
         "policy_mapping_fn": lambda i: 'policy_0'
      },
      'model': {
         'custom_model': 'test_model',
       },
   })
   utils.modelSize(trainer.get_policy('policy_0').model)

   #trainer.restore()
   train(trainer)
   #Evaluator(env, trainer).run()


