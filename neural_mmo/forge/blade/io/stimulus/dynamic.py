from pdb import set_trace as T
import numpy as np
import time

from collections import defaultdict, deque

from neural_mmo.forge.blade.io.stimulus.static import Stimulus as Static
from neural_mmo.forge.blade.entity import Player

def camel(string):
   '''Convert a string to camel case'''
   return string[0].lower() + string[1:]

class Stimulus:
   '''Static IO class used for interacting with game observations

   The environment returns game objects in observations.
   This class assembles them into usable data packets'''
   def process(config, env, ent):
      '''Utility for preprocessing game observations

      Built to be semi-automatic and only require small updates
      for large new classes of observations

      Args:
         config    : An environment configuration object
         env       : Local environment observation
         ent       : Local entity observation
      '''

      #Static handles
      functions = [Stimulus.tile, Stimulus.entity]
      static    = [Static.Tile, Static.Entity]

      index = 0
      stim  = {}
      raw   = {}
      ent.stim = defaultdict(list)
      for func, stat in zip(functions, static):
         stim[stat], raw[stat] = func(env, ent, stat, config)

      return stim, raw

   def add(static, obj, config, args):
      '''Pull attributes from game'''
      #Cache observation representation
      if obj.repr is None:
          obj.repr = {}
          for name, attr in static:
             val = obj
             for n in name:
                val = val.__dict__[camel(n)]

             obj.repr[attr] = attr(config)

      stim = {}
      for name, attr in obj.repr.items():
         stim[name] = attr.get(*args)

      return stim

   def nop(template):
      stim = {}
      for attr in template:
         stim[attr] = np.zeros(1)
      return stim

   def tile(env, ent, static, config):
      '''Internal processor for tile objects'''
      stim = []
      raw = []
      for r, row in enumerate(env):
         for c, tile in enumerate(row):
            raw.append(tile)
            args = [tile, r, c]
            s = Stimulus.add(static, tile, config, args)
            s = dict(sorted(s.items()))
            ent.stim[type(tile)].append(tile)
            stim.append(s)
      return stim, raw

   def entity(env, ent, static, config):
      '''Internal processor for player objects. Always returns self first'''
      raw = []
      stim = deque()
      for tile in env.ravel():
         for e in tile.ents.values():
            if type(e) != Player:
               continue
            raw.append(e)
            args = [ent, e]
            s = Stimulus.add(static, e, config, args)
            s = dict(sorted(s.items()))

            ent.stim[type(e)].append(e)
            if ent is e:
               stim.appendleft(s)
            else:
               stim.append(s)

      #Should sort before truncate
      #Remember to also sort raw_obs
      stim = list(stim)[:config.N_AGENT_OBS]
      return stim, raw
