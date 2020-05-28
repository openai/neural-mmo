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

from projekt.realm import Realm, observationSpace, actionSpace
from projekt.policy import Policy
from projekt.rlutils import SanePPOTrainer
from projekt.evaluator import Evaluator
from projekt.config import Config

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
