from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.io.stimulus import Static

class Data:
   def __init__(self, flat=False):
      self.flat = flat
      self.keys = []
   
      if flat:
         self.data = defaultdict(list)
      else:
         self.data = defaultdict(lambda: defaultdict(list))

   def add(self, static, obj, *args):
      for name, attr in static:
         val = getattr(obj, attr.name).get(*args)
         if self.flat:
            self.data[name].append(val)
         else:
            self.data[obj][name] = val
      self.keys.append(obj)

   @property
   def ret(self):
      return self.keys, self.data

class Dynamic:
   def __call__(self, env, ent, flat=False):
      data, static, self.flat = {}, dict(Static), flat
      data['Entity'] = self.entity(env, ent, static['Entity'])
      data['Tile']   = self.tile(env, static['Tile'])
      return data
 
   def tile(self, env, static):
      data = Data(self.flat)
      for r, row in enumerate(env):
         for c, tile in enumerate(row):
            data.add(static, tile, tile, r, c)
      return data.ret

   def entity(self, env, ent, static):
      data = Data(self.flat)
      for tile in env.ravel():
         for e in tile.ents.values():
            data.add(static, e, ent)
      return data.ret


