from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.io.stimulus import Static
from forge.blade.io import utils

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
      static, self.flat = dict(Static), flat

      keys      = 'Entity Tile'.split()
      functions = [self.entity, self.tile]
      
      data = {}
      for key, f in zip(keys, functions):
         data[key]  = f(env, ent, static[key])
      return data

   def batch(stims):
      retKeys = defaultdict(list)
      retVals = defaultdict(lambda: defaultdict(list))

      #Key of main ent, stim
      for stim in stims:
         for stat, stimSet in stim.items():
            #Key by sub ents
            keys, vals = stimSet
            retKeys[stat].append(keys)

            for attr, val in vals.items():
               #Remove an extra dim
               val = np.array(val).reshape(-1)
               retVals[stat][attr].append(val)

      #Separate this shit into serial
      #for group, stat in retKeys.items():
      #   retKeys[group] = utils.pack(stat)

      for group, stat in retVals.items():
         for attr, vals in stat.items():
            vals, keys = utils.pack(vals)
            retVals[group][attr] = vals

      return retKeys, retVals

   def unbatch(retKeys, retVals):
      n = len(retKeys['Entity'])
      stims = [defaultdict(list) for _ in range(n)]

      for group, stat in retKeys.items():
         for idx, keys in enumerate(stat):
            stims[idx][group] = [keys, defaultdict(list)]

      for group, stat in retVals.items():
         for attr, vals in stat.items():
            keys = retKeys[group]
            lens = [len(e) for e in keys]
            vals = utils.unpack(vals, lens)
            for idx, val in enumerate(vals):
               stims[idx][group][1][attr] = val
      return stims
            
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
