from pdb import set_trace as T
import numpy as np
import inspect

from collections import defaultdict
from itertools import chain

from forge.ethyr import node

class Config:
   ROOT = 'resource/maps/procedural/map'
   SUFFIX = '/map.tmx'

   SZ = 62
   BORDER = 9
   R = C = SZ + BORDER

   STIM = 7
   WINDOW = 2*STIM + 1

   NENT = 256
   RESOURCE = 32

   #Attack ranges
   MELEERANGE = 1
   RANGERANGE = 2
   MAGERANGE  = 3

   MELEEDAMAGE = 10
   RANGEDAMAGE = 2
   MAGEDAMAGE  = 1

   def __init__(self, remote=False, **kwargs):
      self.static = Stim
      self.dynamic = Dynamic(Stim)
      for k, v in kwargs.items():
         setattr(self, k, v)

   def SPAWN(self):
      R, C = Config.R, Config.C
      spawn, border, sz = [], Config.BORDER, Config.SZ
      spawn += [(border, border+i) for i in range(sz)]
      spawn += [(border+i, border) for i in range(sz)]
      spawn += [(R-1, border+i) for i in range(sz)]
      spawn += [(border+i, C-1) for i in range(sz)]
      idx = np.random.randint(0, len(spawn))
      return spawn[idx]

class MetaInspect(type):
   def __iter__(cls):
      for name, attr in cls.__dict__.items():
         if not inspect.isclass(attr):
            continue
         yield name, attr

class Dynamic:
   def __init__(self, static):
      self.static = static

   def __call__(self, env, ent, flat=False):
      data, static = {}, dict(self.static)
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

class Stim(metaclass=MetaInspect):
   class Entity(metaclass=MetaInspect):
      class Food(node.Continuous):
         default = max = Config.RESOURCE

      class Water(node.Continuous):
         default = max = Config.RESOURCE

      class Health(node.Continuous):
         default = max = 10

      class TimeAlive(node.Continuous):
         default = 0

      class Damage(node.Continuous):
         default = None

      class Freeze(node.Continuous):
         default = 0

      class Immune(node.Continuous):
         default = max = 15

      class SameColor(node.Discrete):
         max = 1

      class R(node.Discrete):
         max = Config.R

      class C(node.Discrete):
         max = Config.C

      class RDelta(node.Discrete):
         min = -Config.STIM
         max = Config.STIM

      class CDelta(node.Discrete):
         min = -Config.STIM
         max = Config.STIM

   class Tile(metaclass=MetaInspect):
      class Index(node.Discrete):
         max = 6 #Number of tile types

         def get(self, tile, r, c):
            return tile.state.index
   
      class NEnts(node.Continuous):
         max = Config.NENT

         def get(self, tile, r, c):
            return len(tile.ents)
 
      class R(node.Continuous):
         max = Config.WINDOW 

      class C(node.Continuous):
         max = Config.WINDOW 

     

