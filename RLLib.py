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

import ray
from ray import rllib
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune import registry
import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import preprocessors
import argparse

from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.stimulus import node
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
   NENT    = 256
   NPOP    = 8

   NREALM  = 256
   NWORKER = 12

   EMBED   = 32
   HIDDEN  = 32
   STIM    = 4
   WINDOW  = 9
   #WINDOW  = 15

   ENT_OBS = 20

   OPTIM_STEPS = 128
   DEVICE      = 'cpu'

class Policy(TorchModelV2, nn.Module):
   def __init__(self, *args, **kwargs):
      TorchModelV2.__init__(self, *args, **kwargs)
      nn.Module.__init__(self)

      #self.io = IO(Config)
      self.input = io.Input(Config, policy.TaggedInput,
            baseline.Attributes, baseline.Entities)
      self.output = io.Output(Config)
      #self.output = nn.Linear(Config.HIDDEN, 4)
      self.valueF = nn.Linear(Config.HIDDEN, 1)

   def forward(self, input_dict, state, seq_lens):
      obs           = input_dict['obs']
      state, lookup = self.input(obs)
      atns          = self.output(state, lookup)

      from forge.blade.io.action import static
      atns =  atns[static.Move][static.Direction]
      #atns          = self.output(state)
      self.value    = self.valueF(state).squeeze(1)
      return atns, []

   def value_function(self):
      return self.value

class Realm(core.Realm, rllib.MultiAgentEnv):
   def __init__(self, config, idx=0):
      super().__init__(config, idx)
      self.lifetimes = []
           
   def reset(self):
      #assert self.tick == 0
      return self.step({})[0]

   def preprocess(env, ent):
      pass
      
      
   '''Example environment overrides'''
   def step(self, decisions):
      #Postprocess actions
      for entID, atn in decisions.items():
         arg = StaticAction.Direction.edges[atn]
         decisions[entID] = {StaticAction.Move: [arg]}

      stims, rewards, dones, _ = super().step(decisions)
      preprocessed        = {}
      preprocessedRewards = {}
      preprocessedDones   = {}
      for idx, stim in enumerate(stims):
         env, ent = stim
         conf = self.config
         n = conf.STIM
         s = []

         obs = stimulus.Dynamic.process(self.config, env, ent, True)

         preprocessed[ent.entID] = obs
         preprocessedRewards[ent.entID] = 0
         preprocessedDones[ent.entID]   = False
         preprocessedDones['__all__']   = False #self.tick % 200 == 0

      for ent in dones:
         #Why is there a final obs? Zero this?
         preprocessed[ent.entID]        = obs
         preprocessedDones[ent.entID]   = True
         preprocessedRewards[ent.entID] = -1

         self.lifetimes.append(ent.history.timeAlive.val)
         if len(self.lifetimes) >= 1000:
            lifetime = np.mean(self.lifetimes)
            self.lifetimes = []
            print('Lifetime: {}'.format(lifetime))

      return preprocessed, preprocessedRewards, preprocessedDones, {}

def env_creator(args):
   return Realm(Config(), idx=0)

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

   obs  = spaces.Dict(obs)
   atns = gym.spaces.Discrete(4)

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

   def tick(self):
      atns = {}
      for agentID, ob in self.obs.items():
         if agentID in self.done and self.done[agentID]:
            continue
         atns[agentID] = trainer.compute_action(self.obs[agentID],
               policy_id='policy_{}'.format(0))

      self.obs, rewards, self.done, _ = self.env.step(atns)
 
if __name__ == '__main__':
   #lib.ray.init(Config, 'local')
   ray.init(local_mode=False)

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
      'sample_batch_size': 100,
      'train_batch_size': 4000,
      'sgd_minibatch_size': 128,
      'num_sgd_iter': 1,
      'use_pytorch': True,
      'horizon': np.inf,
      'soft_horizon': True, 
      "multiagent": {
         "policies": policies,
         "policy_mapping_fn": ray.tune.function(lambda i: 'policy_0')
      },
      'model': {
         'custom_model': 'test_model',
       },
   })
   utils.modelSize(trainer.get_policy('policy_0').model)

   #trainer.restore()
   train(trainer)
   #Evaluator(env, trainer).run()


