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
   '''Test-time evaluation with communication to
   the Unity3D client. Makes use of batched GPU inference'''
   def __init__(self, config):
      self.config  = config
      self.done    = {}

   def render(self):
      '''Rendering launches a Twisted WebSocket server with a fixed
      tick rate. This is a blocking call; the server will handle 
      environment execution using the provided tick function.'''
      from forge.trinity.twistedserver import Application
      Application(self.env, self.tick)

   def test(self):
      for t in tqdm(range(self.config.EVALUATION_HORIZON)):
         self.tick(None, None)

      log = InkWell()
      log.update(self.env.terminal())

      np.save(self.config.PATH_EVAL_DATA, log.packet)

   def tick(self, actions, preprocessActions=True):
      '''Compute actions and overlays for a single timestep
      Args:
          pos: Camera position (r, c) from the server)
          cmd: Console command from the server
      '''
      self.obs, rewards, self.done, _ = self.env.step(
            actions, omitDead=True, preprocessActions=preprocessActions)

class Evaluator(Base):
   def __init__(self, config, policy):
      super().__init__(config)
      self.policy   = policy

      self.env      = Env(config)
      self.obs      = self.env.reset()
      self.registry = OverlayRegistry(config, self.env).init()

   def tick(self, pos, cmd):
      realm, actions    = self.env.realm, {}
      for agentID in self.obs:
         agent              = realm.players[agentID]
         agent.skills.style = Action.Range
         actions[agentID]   = self.policy(realm, agent)

      self.registry.step(self.obs, pos, cmd)
      super().tick(actions, preprocessActions=False)
