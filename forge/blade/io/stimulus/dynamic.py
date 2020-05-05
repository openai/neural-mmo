from pdb import set_trace as T
import numpy as np
import time

from collections import defaultdict, deque

from forge.blade.io.stimulus.static import Stimulus as Static
from forge.blade.io.serial import Serial

def camel(string):
   '''Convert a string to camel case'''
   return string[0].lower() + string[1:]

class Stimulus:
   '''Static IO class used for interacting with game observations

   The environment returns game objects in observations.
   This class assembles them into usable data packets'''
   def process(config, env, ent, serialize=True):
      '''Utility for preprocessing game observations

      Built to be semi-automatic and only require small updates
      for large new classes of observations

      Args:
         config    : An environment configuration object
         inp       : An IO object specifying observations
         env       : Local environment observation
         ent       : Local entity observation
         serialize : (bool) Whether to serialize the IO object data
      '''

      #Static handles
      Stimulus.functions = [Stimulus.tile, Stimulus.entity]
      Stimulus.static    = dict(Static)

      index = 0
      stim  = {}
      raw   = {}
      ent.stim = defaultdict(list)
      keys = list(Stimulus.static.keys())
      for key, f in zip(keys, Stimulus.functions):
         stim[key], raw[key] = f(env, ent, key, config)

      return stim, raw

   def add(static, obj, *args, key):
      '''Pull attributes from game and serialize names'''
      stim = {}
      for name, attr in static:
         val = obj
         for n in name:
            val = val.__dict__[camel(n)]

         val = val.get(*args)
         stim[name] = val

      return stim

   def nop(static):
      stim = {}
      for name, attr in static:
         stim[name] = np.array([0]) 
      return stim

   def tile(env, ent, key, config):
      '''Internal processor for tile objects'''
      stim = []
      raw = []
      static = Stimulus.static[key]
      for r, row in enumerate(env):
         for c, tile in enumerate(row):
            raw.append(tile)
            s = Stimulus.add(static, tile, tile, r, c,
                  key=ent)
            ent.stim[type(tile)].append(tile)
            stim.append(s)
      return stim, raw

   def entity(env, ent, key, config):
      '''Internal processor for player objects. Always returns self first'''
      raw = []
      static, stim = Stimulus.static[key], deque()
      for tile in env.ravel():
         for e in tile.ents.values():
            raw.append(e)
            s = Stimulus.add(static, e, ent, e, key=ent)
            ent.stim[type(e)].append(e)
            if ent is e:
               stim.appendleft(s)
            else:
               stim.append(s)

      stim = list(stim)[:config.ENT_OBS]
      nop = Stimulus.nop(static)
      while len(stim) < config.ENT_OBS:
         stim.append(nop)

      return stim, raw
