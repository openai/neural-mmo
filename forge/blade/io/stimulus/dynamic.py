from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.io.stimulus import Static

class Data:
   def __init__(self, flat=False):
      self.flat = flat
      self.keys = []
      self.key = None
   
      if flat:
         self.data = defaultdict(list)
      else:
         self.data = defaultdict(lambda: defaultdict(list))

   def add(self, static, obj, *args, key):
      for name, attr in static:
         val = getattr(obj, attr.name).get(*args)
         if self.flat:
            if key != self.key:
               self.data[name].append([])
            self.data[name][-1].append(val)
         else:
            if key != self.key:
               self.data[obj][name].append([])
            self.data[obj][name] = val
      self.key = key
      self.keys.append(obj)

   @property
   def ret(self):
      return self.keys, self.data

class Dynamic:
   def __call__(self, stim, flat=False):
      env, ent = stim
      data, static, self.flat = {}, dict(Static), flat
      data['Entity'] = self.entity(env, ent, static['Entity'])
      data['Tile']   = self.tile(env, ent, static['Tile'])
      return data

   def merge(stims):
      retKeys = defaultdict(list)
      retVals = defaultdict(lambda: defaultdict(list))

      for stim in stims:
         for stat, stimSet in stim.items():
            keys, vals = stimSet
            retKeys[stat] += keys
            for name, attr in vals.items():
               retVals[stat][name] += attr
      return retKeys, retVals

   def tile(self, env, ent, static):
      data = Data(self.flat)
      for r, row in enumerate(env):
         for c, tile in enumerate(row):
            data.add(static, tile, tile, r, c, key=ent)
      return data.ret

   def entity(self, env, ent, static):
      data = Data(self.flat)
      for tile in env.ravel():
         for e in tile.ents.values():
            data.add(static, e, ent, e, key=ent)
      return data.ret
