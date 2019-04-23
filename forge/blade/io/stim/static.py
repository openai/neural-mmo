from pdb import set_trace as T
import numpy as np
import inspect

from collections import defaultdict
from itertools import chain

from forge.blade.core.config import Config
from forge.blade.io.stim import node

class MetaInspect(type):
   def __iter__(cls):
      for name, attr in cls.__dict__.items():
         if not inspect.isclass(attr):
            continue
         yield name, attr

class Static(metaclass=MetaInspect):
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

     

