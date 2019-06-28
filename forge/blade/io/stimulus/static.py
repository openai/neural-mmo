from pdb import set_trace as T
import numpy as np

import inspect

from forge.blade.io.stimulus import node

class InnerClassIterable(type):
   def __iter__(cls):
      for name, attr in cls.__dict__.items():
         if not inspect.isclass(attr):
            continue
         yield name, attr

class Config(metaclass=InnerClassIterable):
   pass

class Stimulus(Config):
   class Entity(Config):
      class Food(node.Continuous):
         def init(self, config):
            self.default = config.RESOURCE
            self.max     = config.RESOURCE

      class Water(node.Continuous):
         def init(self, config):
            self.default = config.RESOURCE
            self.max     = config.RESOURCE

      class Health(node.Continuous):
         def init(self, config):
            self.default = config.HEALTH 
            self.max     = config.HEALTH

      class TimeAlive(node.Continuous):
         def init(self, config):
            self.default = 0

      class Damage(node.Continuous):
         def init(self, config):
            self.default = None

      class Freeze(node.Continuous):
         def init(self, config):
            self.default = 0

      class Immune(node.Continuous):
         def init(self, config):
            self.default = config.IMMUNE
            self.max     = config.IMMUNE

      class Self(node.Discrete):
         def init(self, config):
            self.default = 0
            self.max = 1

         def get(self, ent, ref):
            val = int(ent is ref)
            return self.asserts(val)
 
      class Population(node.Discrete):
         def init(self, config):
            self.default = None
            self.max = config.NPOP

      class R(node.Discrete):
         def init(self, config):
            self.min = -config.STIM
            self.max = config.STIM

         def get(self, ent, ref):
            val = self.val - ref.r.val
            return self.asserts(val)
 
      class C(node.Discrete):
         def init(self, config):
            self.min = -config.STIM
            self.max = config.STIM

         def get(self, ent, ref):
            val = self.val - ref.c.val
            return self.asserts(val)

   class Tile(Config):
      #A multiplicative interaction between pos and index
      #is required at small training scale
      #class PosIndex(node.Discrete):
      #   def init(self, config):
      #      self.max = config.NTILE*15*15

      #   def get(self, tile, r, c):
      #      return (r*15+c)*tile.state.index
     
      class NEnts(node.Continuous):
         def init(self, config):
            self.max = config.NENT

         def get(self, tile, r, c):
            return len(tile.ents)

      class Index(node.Discrete):
         def init(self, config):
            self.max = config.NTILE

         def get(self, tile, r, c):
            return tile.state.index

      '''
      class Position(node.Discrete):
         def init(self, config):
            self.max = 9

         def get(self, tile, r, c):
            return r*3+c
      '''
 
      class RRel(node.Discrete):
         def init(self, config):
            self.max = config.WINDOW

         def get(self, tile, r, c):
            return r
 
      class CRel(node.Discrete):
         def init(self, config):
            self.max = config.WINDOW

         def get(self, tile, r, c):
            return c

