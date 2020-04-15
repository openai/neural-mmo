from pdb import set_trace as T
import numpy as np
import time

from collections import defaultdict

from forge.blade.io.stimulus.static import Stimulus as Static
from forge.blade.io.serial import Serial

def camel(string):
   '''Convert a string to camel case'''
   return string[0].lower() + string[1:]

'''Internal datastructure used in Stimulus processing
class Data:
   def __init__(self):
      self.keys = []
      self.key = None
   
      self.data = defaultdict(list)

   def add(self, static, obj, *args, key):
      for name, attr in static:
         val = obj
         for n in name:
            val = val.__dict__[n[0].lower() + n[1:]]
         val = val.get(*args)
         if key != self.key:
            self.data[name].append([])
         self.data[name][-1].append(val)
      self.key = key
      self.keys.append(obj)

   @property
   def ret(self):
      return self.keys, self.data
'''

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

      stim = {}
      keys = list(Stimulus.static.keys())
      for key, f in zip(keys, Stimulus.functions):
         stim[key] = f(env, ent, key, serialize)

      return stim

   def add(static, obj, *args, key, serialize=False):
      '''Pull attributes from game and serialize names'''
      stim = {}
      for name, attr in static:
         val = obj
         for n in name:
            val = val.__dict__[camel(n)]
         val = val.get(*args)
         stim[name] = val

      return stim

   def tile(env, ent, key, serialize=False):
      '''Internal processor for tile objects'''
      stim = []
      static = Stimulus.static[key]
      for r, row in enumerate(env):
         for c, tile in enumerate(row):
            s = Stimulus.add(static, tile, tile, r, c,
                  key=ent, serialize=serialize)
            stim.append(s)
      return stim

   def entity(env, ent, key, serialize=False):
      '''Internal processor for player objects. Always returns self first'''
      ents = []
      static = Stimulus.static[key]
      for tile in env.ravel():
         for e in tile.ents.values():
            ents.append(e)
     
      ents = sorted(ents, key=lambda e: e is ent, reverse=True)

      stim = []
      for e in ents:
         s = Stimulus.add(static, e, ent, e, key=ent, serialize=serialize)
         stim.append(s)

      nop = s
      for key, val in s.items():
         if type(val) == np.ndarray:
            nop[key] = np.array([0])
         else:
            nop[key] = 0

      while len(stim) < 20:
         stim.append(nop)

      return stim

      '''
      ents = []
      static = Stimulus.static[key]
      for tile in env.ravel():
         for e in tile.ents.values():
            Stimulus.add(inp, inp.obs.entities[key], inp.lookup,
               static, e, ent, e, key=ent, serialize=serialize)
      '''



