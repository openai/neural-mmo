from pdb import set_trace as T
import numpy as np

from collections import defaultdict
import gym

from ray import rllib

from ray.rllib.models.extra_spaces import Repeated

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

         ents = self.raw[entID][Stimulus.Entity]
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

      obs, rewards, dones, infos = super().step(actions, values, attns)

      for ent in self.dead:
         lifetime = ent.history.timeAlive.val
         self.lifetimes.append(lifetime)
         infos[ent.entID] = {'lifetime': lifetime}
         if not self.config.RENDER and len(self.lifetimes) >= 1000:
            lifetime = np.mean(self.lifetimes)
            print('Lifetime: {}'.format(lifetime))
            dones['__all__'] = True

      return obs, rewards, dones, infos

#To be integrated into RLlib
class FlexDict(gym.spaces.Dict):
   """Gym Dictionary with arbitrary keys"""
   def __init__(self, spaces=None, **spaces_kwargs):
      err = 'Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)'
      assert (spaces is None) or (not spaces_kwargs), err

      if spaces is None:
         spaces = spaces_kwargs

      self.spaces = spaces
      for space in spaces.values():
         self.assertSpace(space)

      # None for shape and dtype, since it'll require special handling
      self.np_random = None
      self.shape     = None
      self.dtype     = None
      self.seed()

   def assertSpace(self, space):
      err = 'Values of the dict should be instances of gym.Space'
      assert isinstance(space, gym.spaces.Space), err

   def sample(self):
      return dict([(k, space.sample())
            for k, space in self.spaces.items()])

   def __getitem__(self, key):
      return self.spaces[key]

   def __setitem__(self, key, space):
      self.assertSpace(space)
      self.spaces[key] = space

   def __repr__(self):
      return "FlexDict(" + ", ". join([str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"

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

