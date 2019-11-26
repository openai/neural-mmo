from pdb import set_trace as T
import numpy as np

import inspect

from forge.blade.io.stimulus import node

#Makes private attributes read only
class InnerClassIterable(type):
   def __iter__(cls):
      stack = list(cls.__dict__.items())
      while len(stack) > 0:
         name, attr = stack.pop()
         if type(name) != tuple:
            name = tuple([name])
         if not inspect.isclass(attr):
            continue
         if issubclass(attr, node.Flat):
            for n, a in attr.__dict__.items():
               n = name + tuple([n]) 
               stack.append((n, a))
            continue
         yield name, attr

class Config(metaclass=InnerClassIterable):
   pass

class StimHook:
   def __init__(self, meta, config):
      self.meta = meta
      self.config = config
   
      self.inputs(meta, config)

   def inputs(self, cls, config):
      for name, c in cls:
         self.__dict__[c.name] = c(config)

   def outputs(self, config):
      data = {}
      for name, cls in self.meta:
         assert type(name) == tuple and len(name) == 1
         name       = name[0].lower()
         attr       = self.__dict__[cls.name]
         data[name] = attr.packet()

      return data

   def packet(self):
      return self.outputs(self.config)


class Stimulus(Config):
   def dict():
      return { k[0] : v for k, v in dict(Stimulus).items()}

   class Entity(Config):
      #Base data
      class Base(Config, node.Flat):
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
               val = self.val - ref.base.r.val
               return self.asserts(val)
    
         class C(node.Discrete):
            def init(self, config):
               self.min = -config.STIM
               self.max = config.STIM

            def get(self, ent, ref):
               val = self.val - ref.base.c.val
               return self.asserts(val)

      #Historical stats
      class History(Config, node.Flat):
         class Damage(node.Continuous):
            def init(self, config):
               self.default = None
               self.scale = 0.01

         class TimeAlive(node.Continuous):
            def init(self, config):
               self.default = 0
               self.scale = 0.01

      #Resources
      class Resources(Config, node.Flat):
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

      #Status effects
      class Status(Config, node.Flat):
         class Freeze(node.Continuous):
            def init(self, config):
               self.default = 0
               self.max     = 3

         class Immune(node.Continuous):
            def init(self, config):
               self.default = config.IMMUNE
               self.max     = config.IMMUNE

         class Wilderness(node.Continuous):
            def init(self, config):
               self.default = -1
               self.min     = -1
               self.max     = 126

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

