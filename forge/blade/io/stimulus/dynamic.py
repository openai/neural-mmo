from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.io.stimulus import Static

class Dynamic:
   def __call__(self, env, ent, flat=False):
      data, static = {}, dict(Static)
      data['Entity'] = self.entity(env, ent, static['Entity'], flat)
      data['Tile']   = self.tile(env, static['Tile'], flat)
      return data

   def add(self, data, static, obj, *args, flat=True):
      if not flat:
         data[obj] = {}
      for name, attr in static:
         val = getattr(obj, attr.name).get(*args)
         if flat:
            data[name].append(val)
         else:
            data[obj][name] = val
            
 
   def tile(self, env, static, flat):
      data = defaultdict(list)
      for r, row in enumerate(env):
         for c, tile in enumerate(row):
            self.add(data, static, tile, tile, r, c, flat=flat)
      return data

   def entity(self, env, ent, static, flat):
      data = defaultdict(list)
      for tile in env.ravel():
         for e in tile.ents.values():
            self.add(data, static, e, ent, flat=flat)
      return data


