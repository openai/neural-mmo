from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from projekt.overlay import Overlays

class Evaluator:
   '''Test-time evaluation with communication to
   the Unity3D client. Makes use of batched GPU inference'''
   def __init__(self, trainer, env, config):
      self.obs   = env.reset(idx=0)
      self.env   = env

      self.state = {}
      self.done  = {}

      self.config   = config
      config.RENDER = True

      self.trainer  = trainer
      self.model    = self.trainer.get_policy('policy_0').model
      self.overlays = Overlays(env, self.model, trainer, config)

   def run(self):
      '''Rendering launches a Twisted WebSocket server with a fixed
      tick rate. This is a blocking call; the server will handle 
      environment execution using the provided tick function.'''
      from forge.trinity.twistedserver import Application
      Application(self.env, self.tick)

   def tick(self):
      '''#Compute actions and overlays for a single timestep'''

      #Remove dead agents
      for agentID in self.done:
         if self.done[agentID]:
            del self.obs[agentID]
           
      #Compute batch of actions
      actions, self.state, _ = self.trainer.compute_actions(
            self.obs, state=self.state, policy_id='policy_0')

      #Compute overlay maps
      self.overlays.register(self.obs)

      #Step the environment
      self.obs, rewards, self.done, _ = self.env.step(actions)
