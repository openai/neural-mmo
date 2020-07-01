from pdb import set_trace as T
import numpy as np

from collections import defaultdict
import gym

from ray import rllib

from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.spaces.flexdict import FlexDict

from forge.blade import core
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.action.static import Action

class Realm(core.Realm, rllib.MultiAgentEnv):
   def __init__(self, config):
      self.config = config['config']

   def reset(self, idx=None):
      '''Enable us to reset the Neural MMO environment.
      This is for training on limited-resource systems where
      simply using one env map per core is not feasible'''
      n   = self.config.NMAPS
      if idx is None:
         idx = np.random.randint(n)

      self.lifetimes = []
      super().__init__(self.config, idx)
      return self.step({})[0]

   def step(self, decisions):
      '''Action postprocessing; small wrapper to fit RLlib'''
      actions = {}
      for entID in list(decisions.keys()):
         actions[entID] = defaultdict(dict)
         if entID in self.dead:
            continue

         ents = self.raw[entID][Stimulus.Entity]
         for atn, args in decisions[entID].items():
            for arg, val in args.items():
               val = int(val)
               if len(arg.edges) > 0:
                  actions[entID][atn][arg] = arg.edges[val]
               elif val < len(ents):
                  actions[entID][atn][arg] = ents[val]
               else:
                  actions[entID][atn][arg] = ents[0]

      obs, rewards, dones, infos = super().step(actions)

      #Cull dead agaents
      for ent in self.dead:
         lifetime = ent.history.timeAlive.val
         self.lifetimes.append(lifetime)
         infos[ent.entID] = {'lifetime': lifetime}
         if not self.config.RENDER and len(self.lifetimes) >= 1000:
            lifetime = np.mean(self.lifetimes)
            print('Lifetime: {}'.format(lifetime))
            dones['__all__'] = True

      return obs, rewards, dones, infos

#Neural MMO observation space
def observationSpace(config):
   obs = FlexDict({})
   for entity in sorted(Stimulus.values()):
      attrDict = FlexDict({})
      for attr in sorted(entity.values()):
         attrDict[attr] = attr(config).space
      n           = entity.N(config)
      obs[entity] = Repeated(attrDict, max_len=n)
   return obs

#Neural MMO action space
def actionSpace(config):
   atns = FlexDict(defaultdict(FlexDict))
   for atn in sorted(Action.edges):
      for arg in sorted(atn.edges):
         n              = arg.N(config)
         atns[atn][arg] = gym.spaces.Discrete(n)
   return atns

