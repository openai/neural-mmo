'''Main file for the neural-mmo/projekt demo

/projeckt contains all necessary RLlib wrappers to train and
evaluate capable policies on Neural MMO as well as rendering,
logging, and visualization tools.

Associated docs and tutorials are hosted on jsuarez5341.github.io.'''
from pdb import set_trace as T

import numpy as np

from fire import Fire

from forge.ethyr.torch import utils
from forge.trinity.scripted import baselines
from forge.trinity.visualize import BokehServer
from forge.trinity.evaluator import Evaluator

import projekt
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
   return wrapper.SanePPOTrainer(config={
      'num_workers': config.NUM_WORKERS,
      'num_gpus_per_worker': config.NUM_GPUS_PER_WORKER,
      'num_gpus': config.NUM_GPUS,
      'num_envs_per_worker': 1,
      'train_batch_size': config.TRAIN_BATCH_SIZE // 2,
      'rollout_fragment_length': config.ROLLOUT_FRAGMENT_LENGTH,
      'sgd_minibatch_size': config.SGD_MINIBATCH_SIZE,
      'num_sgd_iter': config.NUM_SGD_ITER,
      'framework': 'torch',
      'horizon': np.inf,
      'soft_horizon': False, 
      'no_done_at_end': False,
      'callbacks': wrapper.RLlibLogCallbacks,
      'env_config': {
         'config': config
      },
      'multiagent': {
         'policies': policies,
         'policy_mapping_fn': mapPolicy,
         'count_steps_by': 'env_steps'
      },
      'model': {
         'custom_model': 'godsword',
         'custom_model_config': {'config': config},
         'max_seq_len': config.LSTM_BPTT_HORIZON
      },
   })

def loadEvaluator(config):
   '''Create test/render evaluator'''
   if config.SCRIPTED:
       return Evaluator(config, getattr(baselines, config.SCRIPTED))
   else:
      return wrapper.RLlibEvaluator(config, loadModel(config))

def loadModel(config):
   '''Load NN weights and optimizer state'''
   trainer = loadTrainer(config)
   utils.modelSize(trainer.defaultModel())
   if config.LOAD:
      trainer.restore()
   return trainer

class Anvil():
   '''Neural MMO CLI powered by Google Fire

   Main file for the RLlib demo included with Neural MMO.

   Usage:
      python Forge.py <COMMAND> --config=<CONFIG> --ARG1=<ARG1> ...

   The User API documents core env flags. Additional config options specific
   to this demo are available in projekt/config.py. 

   The --config flag may be used to load an entire group of options at once.
   The Debug, SmallMaps, and LargeMaps options are included in this demo with
   the latter being the default -- or write your own in projekt/config.py
   '''
   def __init__(self, **kwargs):
      if 'help' in kwargs:
         kwargs.pop('help')
      if 'config' in kwargs:
         config = kwargs.pop('config')
         config = getattr(projekt.config, config)()
      else:
         config = projekt.config.LargeMaps()
      config.override(**kwargs)
      self.config = config

      if not config.SCRIPTED:
         global torch, ray, rllib, wrapper
         import torch
         import ray
         from ray import rllib
         from projekt import rllib_wrapper as wrapper

   def train(self, **kwargs):
      '''Train a model starting with the current value of --MODEL'''
      loadModel(self.config).train()

   def evaluate(self, **kwargs):
      '''Evaluate a model on --EVAL_MAPS maps'''
      self.config.EVALUATE = True
      loadEvaluator(self.config).evaluate(self.config.GENERALIZE)

   def render(self, **kwargs):
      '''Start a WebSocket server that autoconnects to the 3D Unity client'''
      self.config.RENDER = True
      loadEvaluator(self.config).render()

   def generate(self, **kwargs):
      '''Generate game maps for the current --config setting'''
      terrain.MapGenerator(self.config).generate()

   def visualize(self, **kwargs):
      '''Training/Evaluation results Web dashboard'''
      BokehServer(self.config)
     
if __name__ == '__main__':
   def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)

   from fire import core
   core.Display = Display
   Fire(Anvil)
