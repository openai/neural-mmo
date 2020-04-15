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

from forge.blade import core
from forge.blade.lib.ray import init
from forge.blade.io.action import static as action
from forge.ethyr.torch import param
from forge.ethyr.torch.policy.baseline import IO

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
   WINDOW  = 4

   OPTIM_STEPS = 128
   DEVICE      = 'cpu'

class Preprocessor(preprocessors.NoPreprocessor):
   pass

class Policy(TorchModelV2, nn.Module):
   def __init__(self, *args, **kwargs):
      TorchModelV2.__init__(self, *args, **kwargs)
      nn.Module.__init__(self)

      self.io = IO(Config)

   def forward(self, input_dict, state, seq_lens):
      obs           = input_dict['obs']
      state, lookup = self.io.input(obs)
      atns  = self.io.output(state, lookup)
      return atns

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
         arg = action.Direction.edges[atn]
         decisions[entID] = {action.Move: [arg]}

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
         preprocessedDones['__all__']   = self.tick % 200 == 0

      for ent in dones:
         #Why is there a final obs? Zero this?
         preprocessed[ent.entID]        = obs
         preprocessedDones[ent.entID]   = True
         preprocessedRewards[ent.entID] = -1

         self.lifetimes.append(ent.history.timeAlive.val)
         if len(self.lifetimes) >= 2000:
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
            #attrDict[attr] = spaces.Discrete(val.range)
            attrDict[attr] = spaces.Discrete(100)
         elif issubclass(val, node.Continuous):
            #attrDict[attr] = spaces.Box(low=val.min, high=val.max, shape=(1,))
            attrDict[attr] = spaces.Box(low=-1000, high=1000, shape=(1,))
      entityDict[entity] = spaces.Dict(attrDict)

   key  = tuple(['Tile'])
   obs[key] = spaces.Tuple([entityDict[key] for _ in range(225)])

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
      return super().save(self.saveDir)

   def restore(self):
      path = self.saveDir
      #For some reason, rllib saves checkpoint_idx/checkpoint_idx
      for i in range(2):
         path        = os.path.join(path, '*')
         checkpoints = glob.glob(path)
         path        = max(checkpoints)
      path = path.split('.')[0]
      super().restore(path)

def train(trainer):
   epoch = 0
   while True:
       stats = trainer.train()
       trainer.save()

       nSteps = stats['info']['num_steps_trained']
       nTrajs = -sum(stats['hist_stats']['policy_policy_0_reward'])
       length = nSteps / nTrajs
       print('Epoch: {}, Reward: {}'.format(epoch, length ))

       #if epoch % 5 == 0:
       #   renderRollout()
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
   ray.init(local_mode=True)

   ModelCatalog.register_custom_model('test_model', Policy)
   ModelCatalog.register_custom_preprocessor('test_preprocessor', Preprocessor)
   registry.register_env("custom", env_creator)
   env      = env_creator({})

   policies = {"policy_{}".format(i): gen_policy(env, i) for i in range(1)}
   keys     = list(policies.keys())

   trainer = SanePPOTrainer(env="custom", path='experiment', config={
      'use_pytorch': True,
      'no_done_at_end': True,
      'model': {
         'custom_model': 'test_model',
         #'custom_preprocessor': 'test_preprocessor'
      },
      "multiagent": {
         "policies": policies,
         "policy_mapping_fn": ray.tune.function(lambda i: 'policy_0')
      },
   })

   #trainer.restore()
   train(trainer)
   #Evaluator(env, trainer).run()


