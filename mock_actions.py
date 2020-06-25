from pdb import set_trace as T

from collections import Counter
import gym
import numpy as np
import os
import random
import time
import unittest

import ray
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.test_utils import check, framework_iterator
from ray import rllib
from ray.tune.registry import register_env
from ray.tune import registry

from ray.rllib.models import extra_spaces

import ray.rllib.agents.ppo.ppo as ppo


class MockPolicy(RandomPolicy):
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        return np.array([random.choice([0, 1])] * len(obs_batch)), [], {}

    def postprocess_trajectory(self,
                               batch,
                               other_agent_batches=None,
                               episode=None):
        assert episode is not None
        return compute_advantages(
            batch, 100.0, 0.9, use_gae=False, use_critic=False)

class MockEnv(rllib.MultiAgentEnv):
    def __init__(self, episode_length, config=None):
        self.episode_length = episode_length
        self.config = config
        self.i      = 0

        self.observation_space = gym.spaces.Dict({
            'obs': gym.spaces.Discrete(1)})
        self.action_space = gym.spaces.Dict({
            'atn': gym.spaces.Discrete(2)})

    def reset(self):
        self.i = 0
        return {
            0: {'obs': self.i},
            1: {'obs': self.i}}

    def step(self, action):
        self.i += 1
        return 0, 1, self.i >= self.episode_length, {}

#Generate RLlib policy
def gen_policy(i):
   obs  = gym.spaces.Dict({'obs': gym.spaces.Discrete(1)})
   atns = gym.spaces.Dict({'atn': gym.spaces.Discrete(2)})

   params = {
               "agent_id": i,
               "obs_space_dict": obs,
               "act_space_dict": atns
            }

   return (None, obs, atns, params)

def env_creator(config={}):
   return MockEnv(10, config)

if __name__ == '__main__':
   ray.init()

   registry.register_env('custom', env_creator)
   policyMap = lambda i: 'policy_0'
   policies  = {'policy_0': gen_policy(0)}

   config = {
      'num_workers': 1,
      'use_pytorch': True,
      'multiagent': {
         'policies': policies,
         'policy_mapping_fn': policyMap
         }
      } 

   trainer = ppo.PPOTrainer(env='custom', config=config)

   env = env_creator()
   obs = env.reset()

   #Fails: pass entire obs
   #trainer.compute_action(obs, policy_id='policy_0')

   #Works: pass agent obs
   trainer.compute_action(obs[0], policy_id='policy_0')
