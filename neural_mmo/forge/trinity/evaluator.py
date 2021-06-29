import numpy as np

from collections import defaultdict
from tqdm import tqdm

from neural_mmo.forge.trinity import Env
from neural_mmo.forge.trinity.overlay import OverlayRegistry

from neural_mmo.forge.blade.io.action import static as Action
from neural_mmo.forge.blade.lib.log import InkWell

class Base:
   '''Base class for test-time evaluators'''
   def __init__(self, config):
      self.config  = config
      self.done    = {}

   def render(self):
      '''Rendering launches a Twisted WebSocket server with a fixed
      tick rate. This is a blocking call; the server will handle 
      environment execution using the provided tick function.'''
      from neural_mmo.forge.trinity.twistedserver import Application
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

   def tick(self, obs, actions, pos, cmd, preprocess=set(), omitDead=True):
      '''Simulate a single timestep

      Args:
          obs: dict of agent observations
          actions: dict of policy actions
          pos: Camera position (r, c) from the server
          cmd: Console command from the server
          preprocessActions: Required for actions provided as indices
      '''
      self.obs, rewards, self.done, _ = self.env.step(
            actions, preprocess, omitDead)
      if self.config.RENDER:
         self.registry.step(obs, pos, cmd)

class Evaluator(Base):
   '''Evaluator for scripted models'''
   def __init__(self, config, policy, *args):
      super().__init__(config)
      self.policies = defaultdict(lambda: policy(config)) 
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
         agent  = realm.players[agentID]
         policy = self.policies[agentID]
         ob     = self.obs[agentID]

         actions[agentID] = policy(ob, *self.args)
         if Action.Attack in actions[agentID]:
            targID = actions[agentID][Action.Attack][Action.Target]
            actions[agentID][Action.Attack][Action.Target] = realm.entity(targID)
         #actions[agentID]   = self.policy(realm, agent, *self.args)

      super().tick(self.obs, actions, pos, cmd)
