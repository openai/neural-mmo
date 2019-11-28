from pdb import set_trace as T
import numpy as np
import time

from collections import defaultdict

from forge.blade.io import Stimulus as Static
from forge.ethyr.io import utils
from forge.ethyr.io.serial import Serial

def camel(string):
   return string[0].lower() + string[1:]

class Data:
   '''Internal datastructure used in Stimulus processing'''
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

class Stimulus:
   '''Static IO class used for interacting with game observations

   The environment returns game objects in observations; this class
   assembles them into usable data packets
   '''
   def process(inp, env, ent, config, serialize=True):
      '''Main utility for processing game observations.

      Built to be semi-automatic and only require small updates
      for large new classes of observations

      Args:
         A list of observations as returned by the environment
      
      Returns:
         An unbatched list of proccessed stimulus packets
      '''
      #Static handles
      Stimulus.funcNames = 'Entity Tile'.split()
      Stimulus.functions = [Stimulus.entity, Stimulus.tile]
      Stimulus.static  = Static.dict()

      for key, f in zip(Stimulus.funcNames, Stimulus.functions):
         f(inp, env, ent, key, serialize)

      inp.obs.n += 1

   def add(inp, obs, lookup, static, obj, *args, key, serialize=False):
      '''Pull attributes from game and serialize names'''
      for name, attr in static:
         val = obj
         for n in name:
            val = val.__dict__[camel(n)]
         val = val.get(*args)
         obs.attributes[name].append(val)

      #Serialize names
      lookupKey = obj
      if serialize:
         objKey = Serial.key(obj)
         key = Serial.key(key)
         lookupKey = key + objKey

      idx = lookup.add(lookupKey, orig=obj)
      inp.obs.names[key].append(idx)

   def tile(inp, env, ent, key, serialize=False):
      '''Internal processor for tile objects'''
      static = Stimulus.static[key]
      for r, row in enumerate(env):
         for c, tile in enumerate(row):
            Stimulus.add(inp, inp.obs.entities[key], inp.lookup,
               static, tile, tile, r, c, key=ent, serialize=serialize)

   def entity(inp, env, ent, key, serialize=False):
      '''Internal processor for player objects. Always returns self first'''
      ents = []
      static = Stimulus.static[key]
      for tile in env.ravel():
         for e in tile.ents.values():
            Stimulus.add(inp, inp.obs.entities[key], inp.lookup,
               static, e, ent, e, key=ent, serialize=serialize)
