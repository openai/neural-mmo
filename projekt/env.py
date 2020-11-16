from pdb import set_trace as T
import numpy as np

import time
from collections import defaultdict
import gym

import time

from ray import rllib

from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.spaces.flexdict import FlexDict

from forge.blade import core
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.action.static import Action
from forge.blade.systems import combat

from forge.trinity.dataframe import DataType

class Env(core.Env):
   def log(self, quill, ent):
      blob = quill.register('Lifetime', quill.LINE, quill.SCATTER, quill.HISTOGRAM, quill.GANTT)
      blob.log(ent.history.timeAlive.val)

      blob = quill.register('Skill Level', quill.STACKED_AREA, quill.SCATTER, quill.HISTOGRAM, quill.RADAR)
      blob.log(ent.skills.range.level,        'Range')
      blob.log(ent.skills.mage.level,         'Mage')
      blob.log(ent.skills.melee.level,        'Melee')
      blob.log(ent.skills.constitution.level, 'Constitution')
      blob.log(ent.skills.defense.level,      'Defense')
      blob.log(ent.skills.fishing.level,      'Fishing')
      blob.log(ent.skills.hunting.level,      'Hunting')

      blob = quill.register('Equipment', quill.STACKED_AREA, quill.HISTOGRAM)
      blob.log(ent.loadout.chestplate.level, 'Chestplate')
      blob.log(ent.loadout.platelegs.level,  'Platelegs')

      blob = quill.register('Wilderness', quill.HISTOGRAM, quill.SCATTER)
      blob.log(combat.wilderness(self.config, ent.pos))

class RLLibEnv(Env, rllib.MultiAgentEnv):
   def __init__(self, config):
      self.config = config['config']
      #self.perfTick = 0

   def reset(self, idx=None, dry=False):
      '''Enable us to reset the Neural MMO environment.
      This is for training on limited-resource systems where
      simply using one env map per core is not feasible'''
      self.env_reset = time.time()

      n   = self.config.NMAPS
      if idx is None:
         idx = np.random.randint(n)

      self.lifetimes = []
      super().__init__(self.config, idx)

      ret = None
      if not dry:
         ret = self.step({})[0]

      self.env_reset = time.time() - self.env_reset
      return ret

   def step(self, decisions):
      '''Action postprocessing; small wrapper to fit RLlib'''
      #start = time.time()
      self.rllib_compat = time.time()

      actions = {}
      for entID in list(decisions.keys()):
         ent = self.realm.players[entID]
         r, c = ent.pos
         radius = self.config.STIM
         grid  = self.realm.dataframe.data['Entity'].grid
         index = self.realm.dataframe.data['Entity'].index
         cent = grid.data[r, c]
         rows = grid.window(
            r-radius, r+radius+1,
            c-radius, c+radius+1)
         rows.remove(cent)
         rows.insert(0, cent)
         rows = [index.teg(e) for e in rows]
         assert rows[0] == ent.entID

         actions[entID] = defaultdict(dict)
         if entID in self.dead:
            continue

         ents = self.realm.players.entities
         #ents = list(self.realm.players.entities.values())
         for atn, args in decisions[entID].items():
            for arg, val in args.items():
               val = int(val)
               if len(arg.edges) > 0:
                  actions[entID][atn][arg] = arg.edges[val]
               #elif val < len(ents):
               elif val < len(rows):
                  actions[entID][atn][arg] = ents[rows[val]]
                  #actions[entID][atn][arg] = ents[val]
               else:
                  #actions[entID][atn][arg] = ents[0]
                  actions[entID][atn][arg] = ent

      self.rllib_compat = time.time() - self.rllib_compat
      self.env_step     = time.time()

      obs, rewards, dones, infos = super().step(actions)

      self.env_step     = time.time() - self.env_step
      env_post          = time.time()

      #Cull dead agents
      for ent in self.dead:
         lifetime = ent.history.timeAlive.val
         self.lifetimes.append(lifetime)
         if not self.config.RENDER and len(self.lifetimes) >= 1000:
            lifetime = np.mean(self.lifetimes)
            print('Lifetime: {}'.format(lifetime))
            dones['__all__'] = True

      self.env_step += time.time() - env_post
      return obs, rewards, dones, infos

#Neural MMO observation space
def observationSpace(config):
   obs = FlexDict(defaultdict(FlexDict))
   for entity in sorted(Stimulus.values()):
      #attrDict = FlexDict({})
      #for attr in sorted(entity.values()):
      #   attrDict[attr] = attr(config, None).space
      nRows = entity.N(config)

      nContinuous = 0
      nDiscrete   = 0
      for _, attr in entity:
         if attr.DISCRETE:
            nDiscrete += 1
         if attr.CONTINUOUS:
            nContinuous += 1

      obs[entity.__name__]['Continuous'] = gym.spaces.Box(
            low=0, high=2500, shape=(nRows, nContinuous),
            dtype=DataType.CONTINUOUS)

      obs[entity.__name__]['Discrete']   = gym.spaces.Box(
            low=0, high=4096, shape=(nRows, nDiscrete),
            dtype=DataType.DISCRETE)

   obs['Entity']['N']   = gym.spaces.Box(
         low=0, high=config.N_AGENT_OBS, shape=(1,),
         dtype=DataType.DISCRETE)

   return obs

#Neural MMO action space
def actionSpace(config):
   atns = FlexDict(defaultdict(FlexDict))
   for atn in sorted(Action.edges):
      for arg in sorted(atn.edges):
         n              = arg.N(config)
         atns[atn][arg] = gym.spaces.Discrete(n)
   return atns

