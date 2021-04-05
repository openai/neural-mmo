from pdb import set_trace as T
import numpy as np

import os

from collections import defaultdict
from tqdm import tqdm

import projekt

from forge.trinity import Env
from forge.trinity.overlay import OverlayRegistry

from forge.blade.io.action import static as Action
from forge.blade.lib.log import InkWell

class Base:
   '''Base class for test-time evaluators'''
   def __init__(self, config):
      self.config  = config
      self.done    = {}

   def render(self):
      '''Rendering launches a Twisted WebSocket server with a fixed
      tick rate. This is a blocking call; the server will handle 
      environment execution using the provided tick function.'''
      from forge.trinity.twistedserver import Application
      Application(self.env, self.tick)

   def evaluate(self, generalize=True):
      '''Evaluate the model on maps according to config params'''

      config = self.config
      log    = InkWell()

      if generalize:
         maps = range(-1, -config.EVAL_MAPS-1, -1)
      else:
         maps = range(1, config.EVAL_MAPS+1)

      print('Number of evaluation maps: {}'.format(len(maps)))
      for idx in maps:
         self.obs = self.env.reset(idx)
         for t in tqdm(range(config.EVALUATION_HORIZON)):
            self.tick(None, None)

         log.update(self.env.terminal())

      #Save data
      np.save(config.PATH_EVALUATION, log.packet)

   def tick(self, obs, actions, pos, cmd, preprocessActions=True):
      '''Simulate a single timestep

      Args:
          obs: dict of agent observations
          actions: dict of policy actions
          pos: Camera position (r, c) from the server
          cmd: Console command from the server
          preprocessActions: Required for actions provided as indices
      '''
      self.obs, rewards, self.done, _ = self.env.step(
            actions, omitDead=True, preprocessActions=preprocessActions)
      if self.config.RENDER:
         self.registry.step(obs, pos, cmd)

class Evaluator(Base):
   '''Evaluator for scripted models'''
   def __init__(self, config, policy, *args):
      super().__init__(config)
      self.policy   = policy
      self.args     = args

      self.env      = Env(config)

   def render(self):
      '''Render override for scripted models'''
      self.obs      = self.env.reset()
      self.registry = OverlayRegistry(self.config, self.env).init()
      super().render()

   def tick(self, pos, cmd):
      '''Simulate a single timestep

      Args:
          pos: Camera position (r, c) from the server)
          cmd: Console command from the server
      '''
      realm, actions    = self.env.realm, {}
      for agentID in self.obs:
         agent              = realm.players[agentID]
         agent.skills.style = Action.Range
         actions[agentID]   = self.policy(realm, agent, *self.args)

      super().tick(self.obs, actions, pos, cmd, preprocessActions=False)
