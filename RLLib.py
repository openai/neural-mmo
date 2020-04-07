from pdb import set_trace as T
import numpy as np

import os
import gym
import time
from matplotlib import pyplot as plt

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
         print(ent.history.timeAlive.val)

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

def renderRollout():
   nAgents = 2
   done    = False
   obs     = env.reset()
   while not done:
      env.render()
      time.sleep(0.6)

      atns = {}
      for agentID in range(nAgents):
         atns[agentID] = trainer.compute_action(obs[agentID],
               policy_id='policy_{}'.format(agentID))

      obs, rewards, done, _ = env.step(atns)
      done = done['__all__']
   plt.close()

if __name__ == '__main__':
   ray.init(local_mode=False)

   register_env("custom", env_creator)
   env      = env_creator({})

   policies = {"policy_{}".format(i): gen_policy(env, i) for i in range(1)}
   keys     = list(policies.keys())

   trainer = ppo.PPOTrainer(env="custom", config={
      'no_done_at_end': True,
      "multiagent": {
         "policies": policies,
         "policy_mapping_fn": ray.tune.function(lambda i: 'policy_0')
      },
   })

   epoch = 0
   while True:
       stats = trainer.train()
       nSteps = stats['info']['num_steps_trained']
       nTrajs = -sum(stats['hist_stats']['policy_policy_0_reward'])
       length = nSteps / nTrajs
       print('Epoch: {}, Reward: {}'.format(epoch, length ))

       #if epoch % 5 == 0:
       #   renderRollout()
       epoch += 1
