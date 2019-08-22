from pdb import set_trace as T
from collections import defaultdict
import numpy as np 
import ray
import projekt

from forge import trinity
from forge.trinity.timed import runtime

from forge.blade.core import realm

from forge.ethyr.io import Stimulus, Action
from forge.ethyr.experience import RolloutManager
from forge.ethyr.torch import Model, optim

from forge.ethyr.io.io import Output

from copy import deepcopy

#@ray.remote
class Sword(trinity.Sword):
   '''Core level Sword API demo

   This core level rollout worker node runs
   a copy of the environment and all associated
   agents. Multiple Swords return observations, 
   actions, and rewards to each server level 
   optimizer node.'''

   def __init__(self, trin, config, args, idx):
      '''Initializes a model, env, and relevent utilities'''

      super().__init__(trin, config, args, idx)
      self.foo = True;
      config        = deepcopy(config)
      config.DEVICE = 'cpu:0'

      self.config   = config
      self.args     = args
      self.ent      = 0

      self.keys = set()

      self.net     = projekt.ANN(config)
      self.manager = RolloutManager()

   @runtime
   def step(self, obs, packet=None):
      '''Synchronizes weights from upstream and
      collects a fixed amount of experience.'''
      '''Steps the agent and environment

      Processes observations, selects actions, and
      steps the environment to obtain new observations.
      Serializes (obs, action, reward) triplets for
      communication to an upstream optimizer node.'''
      self.net.recvUpdate(packet)
      actions = {}

      #Batch observations and compute forward pass
      self.manager.collectInputs(obs)
      for pop, batch in self.manager.batched():
         keys, stim, atns = batch

         #Run the policy
         atns, atnsIdx, vals = self.net(pop, stim, atns)

         #Clear .backward buffers during test
         if self.config.TEST or self.config.POPOPT:
            #atns are detached in torch/io/action
            atnsIdx = atnsIdx.detach()
            vals    = vals.detach()

         for key, atn, atnIdx, val in zip(keys, atns, atnsIdx, vals):
            out = Output(key, atn, atnIdx, val)
            actions.update(out.action)
            self.manager.collectOutputs([out])
         
      #Compute backward pass and logs from rollout objects
      if self.manager.nUpdates >= self.config.SYNCBATCH:
         rollouts, logs = self.manager.step()

         if self.config.TEST or self.config.POPOPT:
            return actions, None, logs

         optim.backward(rollouts, valWeight=0.25,
            entWeight=self.config.ENTROPY, device=self.config.DEVICE)
         grads = self.net.grads()
         return actions, grads, logs

      return actions, None, None

      '''
      #Lol pytorch... thanks for 4 hour bug.
      #Using larger batch sizes causes .backward errors because
      #the graph is retained over all trajectories in the batch,
      #even though only some of them are finished
 
      #Currently specifying retain_graph. This is not great
      #Need to set batch size 1? and figure out what is being stored
 
      Goal for today: get something training
      Possible hints: 
         .backward does not work without retain graph
         Movement only does not work either
         1 population does not work either (not just batch size issue)
         keeping the same graph while using stale weights (big one?)
      
      Options: 
         Flag and drop partial trajectories from .backward.
         Strip down ethyr libraries and simplify it out
         Implement PBT on Pantheon, with RL as an optional addon
         If all else fails, hammer on .backward until you figure out what is broken (reusing weights?)

      '''
 
