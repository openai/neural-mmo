from pdb import set_trace as T
import numpy as np

from collections import defaultdict
import gym

from ray import rllib
from ray.rllib.models import extra_spaces

from forge.blade import core
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.action.static import Action

class Realm(core.Realm, rllib.MultiAgentEnv):
   def __init__(self, config):
      self.config = config['config']

   def reset(self):
      n   = self.config.NMAPS
      idx = np.random.randint(n)

      self.lifetimes = []
      super().__init__(self.config, idx)
      return self.step({})[0]

   '''Example environment overrides'''
   def step(self, decisions, values={}, attns={}):
      #Postprocess actions
      actions = {}
      for entID in list(decisions.keys()):
         actions[entID] = defaultdict(dict)
         if entID in self.dead:
            continue

         ents = self.raw[entID][('Entity',)]
         for atn, args in decisions[entID].items():
            for arg, val in args.items():
               val = int(val)
               if len(arg.edges) > 0:
                  actions[entID][atn][arg] = arg.edges[val]
               elif val < len(ents):
                  #Note: you are not masking over padded agents yet.
                  #This will make learning attacks very hard
                  actions[entID][atn][arg] = ents[val]
               else:
                  actions[entID][atn][arg] = ents[0]

      obs, rewards, dones, _ = super().step(actions, values, attns)

      for ent in self.dead:
         self.lifetimes.append(ent.history.timeAlive.val)
         if not self.config.RENDER and len(self.lifetimes) >= 1000:
            lifetime = np.mean(self.lifetimes)
            print('Lifetime: {}'.format(lifetime))
            dones['__all__'] = True

      return obs, rewards, dones, {}

#Neural MMO observation space
def observationSpace(config):
   obs = extra_spaces.FlexDict({})
   for entity, attrList in sorted(Stimulus):
      attrDict = extra_spaces.FlexDict({})
      for attr, val in sorted(attrList):
         attrDict[attr] = val(config).space
      n = attrList.N(config)
      obs[entity] = extra_spaces.Repeated(attrDict, max_len=n)
   return obs

#Neural MMO action space
def actionSpace(config):
   atns = extra_spaces.FlexDict(defaultdict(extra_spaces.FlexDict))
   for atn in sorted(Action.edges):
      for arg in sorted(atn.edges):
         n = arg.N(config)
         atns[atn][arg] = gym.spaces.Discrete(n)
   return atns

