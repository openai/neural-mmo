'''Main file for the neural-mmo/projekt demo

/projeckt contains all necessary RLlib wrappers to train and
evaluate capable policies on Neural MMO as well as rendering,
logging, and visualization tools.

Associated docs and tutorials are hosted on jsuarez5341.github.io.'''
from pdb import set_trace as T

import numpy as np
import torch

from fire import Fire

import ray
from ray import rllib

from forge.ethyr.torch import utils
from forge.blade.systems import ai

from forge.trinity.visualize import BokehServer
from forge.trinity.evaluator import Evaluator

import projekt
from projekt import rllib_wrapper as wrapper
from forge.blade.core import terrain

def createPolicies(config, mapPolicy):
   '''Generate RLlib policies'''
   obs      = wrapper.observationSpace(config)
   atns     = wrapper.actionSpace(config)
   policies = {}

   for i in range(config.NPOLICIES):
      params = {
            "agent_id": i,
            "obs_space_dict": obs,
            "act_space_dict": atns}
      key           = mapPolicy(i)
      policies[key] = (None, obs, atns, params)

   return policies

def loadTrainer(config):
   '''Create monolithic RLlib trainer object'''
   torch.set_num_threads(1)
   ray.init(local_mode=config.LOCAL_MODE)

   #Register custom env
   ray.tune.registry.register_env("Neural_MMO",
         lambda config: wrapper.RLlibEnv(config))

   #Create policies
   rllib.models.ModelCatalog.register_custom_model('godsword', wrapper.RLlibPolicy)
   mapPolicy = lambda agentID: 'policy_{}'.format(agentID % config.NPOLICIES)
   policies  = createPolicies(config, mapPolicy)

   #Instantiate monolithic RLlib Trainer object.
   return wrapper.SanePPOTrainer(
      env=config.ENV_NAME, path='experiment', config={
      'num_workers': config.NUM_WORKERS,
      'num_gpus_per_worker': config.NUM_GPUS_PER_WORKER,
      'num_gpus': config.NUM_GPUS,
      'num_envs_per_worker': 1,
      'train_batch_size': config.TRAIN_BATCH_SIZE,
      'rollout_fragment_length': config.ROLLOUT_FRAGMENT_LENGTH,
      'sgd_minibatch_size': config.SGD_MINIBATCH_SIZE,
      'num_sgd_iter': config.NUM_SGD_ITER,
      'framework': 'torch',
      'horizon': np.inf,
      'soft_horizon': False, 
      '_use_trajectory_view_api': False,
      'no_done_at_end': False,
      'callbacks': wrapper.RLlibLogCallbacks,
      'env_config': {
         'config': config
      },
      'multiagent': {
         'policies': policies,
         'policy_mapping_fn': mapPolicy,
         'count_steps_by': 'agent_steps'
      },
      'model': {
         'custom_model': 'godsword',
         'custom_model_config': {'config': config}
      },
   })

def loadEvaluator(config):
   '''Create test/render evaluator'''
   config.EVALUATE = True
   if config.MODEL != 'scripted':
      return wrapper.RLlibEvaluator(config, loadModel(config))

   err = 'SCRIPTED_BACKEND may be either dijkstra or dynamic_programming'
   assert config.SCRIPTED_BACKEND in ('dijkstra', 'dynamic_programming'), err
   if config.SCRIPTED_BACKEND == 'dijkstra':
      backend = ai.behavior.forageDijkstra
   elif config.SCRIPTED_BACKEND == 'dynamic_programming':
      backend = ai.behavior.forageDP

   return Evaluator(config, ai.policy.baseline,
         config.SCRIPTED_EXPLORE, backend)

def loadModel(config):
   '''Load NN weights and optimizer state'''
   trainer = loadTrainer(config)
   utils.modelSize(trainer.defaultModel())
   trainer.restore(config.MODEL)
   return trainer

class Anvil():
   '''Google Fire command parser for Neural MMO'''
   def __init__(self, **kwargs):
      if 'config' in kwargs:
         config = kwargs.pop('config')
         config = getattr(projekt.config, config)()
      else:
         config = projekt.config.LargeMaps()
      config.override(**kwargs)
      self.config = config

   def train(self, **kwargs):
      loadModel(self.config).train()

   def evaluate(self, **kwargs):
      loadEvaluator(self.config).test()

   def render(self, **kwargs):
      loadEvaluator(self.config).render()

   def generate(self, **kwargs):
      terrain.MapGenerator(self.config).generate()

   def visualize(self, **kwargs):
      BokehServer(self.config)
      
if __name__ == '__main__':
   Fire(Anvil)
