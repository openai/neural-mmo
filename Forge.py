'''Main file for the neural-mmo/projekt demo

/projeckt contains all necessary RLlib wrappers to train and
evaluate capable policies on Neural MMO as well as rendering,
logging, and visualization tools.

Associated docs and tutorials are hosted on jsuarez5341.github.io.'''
from pdb import set_trace as T

from fire import Fire
import os
from copy import deepcopy

import numpy as np

import projekt
from projekt import config as base_config

from neural_mmo.forge.blade.io.action.static import Action
from neural_mmo.forge.trinity.scripted import baselines
from neural_mmo.forge.trinity.visualize import BokehServer
from neural_mmo.forge.trinity.evaluator import Evaluator

def trainer_wrapper(trainer):
   def train(self):
      stats = self.train()

   trainer.train = train
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
         config = getattr(base_config, config)()
      else:
         config = projekt.config.LargeMaps()
      config.override(**kwargs)
      self.config = config

      if config.SCRIPTED:
         self.evaluator = Evaluator(config, getattr(baselines, config.SCRIPTED))
         return

      global torch, ray, rllib, tune, wrapper, utils, WandbLoggerCallback, CLIReporter
      from neural_mmo.forge.ethyr.torch import utils
      import torch
      import ray
      from ray import rllib, tune
      from ray.tune import CLIReporter
      from ray.tune.integration.wandb import WandbLoggerCallback
      from projekt import rllib_wrapper as wrapper
      import ray.rllib.agents.ppo.ppo as ppo

      torch.set_num_threads(1)
      os.environ['MKL_NUM_THREADS']     = '1'
      os.environ['OMP_NUM_THREADS']     = '1'
      os.environ['NUMEXPR_NUM_THREADS'] = '1'
 
      ray.init(local_mode=config.LOCAL_MODE)

      #Register custom env
      ray.tune.registry.register_env("Neural_MMO",
            lambda config: wrapper.RLlibEnv(config))

      #Create policies
      rllib.models.ModelCatalog.register_custom_model('godsword', wrapper.RLlibPolicy)
      mapPolicy = lambda agentID : 'policy_{}'.format(agentID % config.NPOLICIES)

      obs  = wrapper.observationSpace(config)
      atns = wrapper.actionSpace(config)

      policies = {}
      for i in range(config.NPOLICIES):
         params = {
               "agent_id": i,
               "obs_space_dict": obs,
               "act_space_dict": atns}
         key           = mapPolicy(i)
         policies[key] = (None, obs, atns, params)

      eval_config = deepcopy(config)
      eval_config.EVALUATE = True
      eval_config.AGENTS   = eval_config.EVAL_AGENTS

      #Create rllib config
      rllib_config={
         'num_workers': config.NUM_WORKERS,
         'num_gpus_per_worker': config.NUM_GPUS_PER_WORKER,
         'num_gpus': config.NUM_GPUS,
         'num_envs_per_worker': 1,
         'train_batch_size': config.TRAIN_BATCH_SIZE // 2, #RLlib bug doubles batch size
         'rollout_fragment_length': config.ROLLOUT_FRAGMENT_LENGTH,
         'sgd_minibatch_size': config.SGD_MINIBATCH_SIZE,
         'num_sgd_iter': config.NUM_SGD_ITER,
         'framework': 'torch',
         'horizon': np.inf,
         'soft_horizon': False, 
         'no_done_at_end': False,
         'env': 'Neural_MMO',
         'env_config': {
            'config': config
         },
         'evaluation_config': {
            'env_config': {
               'config': eval_config
            },
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
         'callbacks': wrapper.RLlibLogCallbacks,
         'evaluation_interval': config.EVALUATION_INTERVAL,
         'evaluation_num_episodes': config.EVALUATION_NUM_EPISODES,
         'evaluation_num_workers': config.EVALUATION_NUM_WORKERS,
         'evaluation_parallel_to_training': config.EVALUATION_PARALLEL,
      }

      self.rllib_config  = rllib_config
      self.trainer_class = wrapper.RLlibTrainer
      #self.trainer_class = ppo.PPOTrainer
      self.trainer       = self.trainer_class(rllib_config)
      self.evaluator     = wrapper.RLlibEvaluator(config, self.trainer)

   def train(self, **kwargs):
      '''Train a model starting with the current value of --MODEL'''
      class ConsoleLog(CLIReporter):
          def report(self, trials, done, *sys_info):
              os.system('cls' if os.name == 'nt' else 'clear') 
              super().report(trials, done, *sys_info)

      config = self.config
      tune.run(self.trainer_class,
         config = self.rllib_config,
         name = config.__class__.__name__,
         verbose = config.LOG_LEVEL,
         stop = {'training_iteration': config.TRAINING_ITERATIONS},
         resume = config.LOAD,
         local_dir = 'experiments',
         keep_checkpoints_num = config.KEEP_CHECKPOINTS_NUM,
         checkpoint_freq = config.CHECKPOINT_FREQ,
         checkpoint_at_end = True,
         trial_dirname_creator = lambda _: self.trainer_class.__name__,
         progress_reporter = ConsoleLog(),
         reuse_actors = True,
         callbacks=[WandbLoggerCallback(
             project = 'NeuralMMO',
             api_key_file = 'wandb_api_key',
             log_config = False)],

         )

   def evaluate(self, **kwargs):
      '''Evaluate a model on --EVAL_MAPS maps'''
      self.config.EVALUATE            = True
      self.config.TRAINING_ITERATIONS = 0
      #self.evaluator.evaluate(self.config.GENERALIZE)
      self.train(**kwargs)

   def render(self, **kwargs):
      '''Start a WebSocket server that autoconnects to the 3D Unity client'''
      self.config.RENDER = True
      self.evaluator.render()

   def generate(self, **kwargs):
      '''Generate game maps for the current --config setting'''
      from neural_mmo.forge.blade.core import terrain
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
