from pdb import set_trace as T
import numpy as np

import os
import gym
import time
from matplotlib import pyplot as plt
import glob

import ray
from ray import rllib
from ray.tune import run_experiments
from ray.tune.registry import register_trainable, register_env
import ray.rllib.agents.ppo.ppo as ppo
import argparse

from forge.blade import core
from forge.blade.lib.ray import init
from forge.blade.io.action import static as action
from forge.ethyr.torch import param

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
   HIDDEN  = 64
   WINDOW  = 4

   OPTIM_STEPS = 128

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

         for i in range(n-conf.WINDOW, n+conf.WINDOW+1):
            for j in range(n-conf.WINDOW, n+conf.WINDOW+1):
               tile = env[i, j]
               s += oneHot(tile.mat.index, 6)

         s.append(ent.resources.health.val / ent.resources.health.max)
         s.append(ent.resources.food.val   / ent.resources.food.max)
         s.append(ent.resources.water.val  / ent.resources.water.max)

         preprocessed[ent.entID] = s
         preprocessedRewards[ent.entID] = 0
         preprocessedDones[ent.entID]   = False
         preprocessedDones['__all__']   = self.tick % 200 == 0

      for ent in dones:
         preprocessed[ent.entID]        = (0*np.array(s)).tolist()
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
   obs    = gym.spaces.Box(
         low=0.0, high=1.0, shape=(489,), dtype=np.float32)
   atns   = gym.spaces.Discrete(4)

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
   ray.init(local_mode=False)

   register_env("custom", env_creator)
   env      = env_creator({})

   policies = {"policy_{}".format(i): gen_policy(env, i) for i in range(1)}
   keys     = list(policies.keys())

   trainer = SanePPOTrainer(env="custom", path='experiment', config={
      'use_pytorch': True,
      'no_done_at_end': True,
      "multiagent": {
         "policies": policies,
         "policy_mapping_fn": ray.tune.function(lambda i: 'policy_0')
      },
   })

   trainer.restore()
   #train(trainer)
   Evaluator(env, trainer).run()


