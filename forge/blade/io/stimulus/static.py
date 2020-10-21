from pdb import set_trace as T
import numpy as np

from forge.blade.io import node

def bind(gameCls):
   def to(ioCls):
      @property
      def GAME_CLS():
         return gameCls

      ioCls.GAME_CLS = GAME_CLS
      #ioCls.GAME_CLS = gameCls
      return ioCls
   return to 

class Config(metaclass=node.IterableTypeCompare):
   pass

class Stimulus(Config):
   def dict():
      return {k[0] : v for k, v in dict(Stimulus).items()}

   class Entity(Config):
      @staticmethod
      def N(config):
         return config.N_AGENT_OBS

      class Self(node.Discrete):
         def init(self, config):
            self.val = 0
            self.max = 1

         def get(self, ent, ref):
            val = int(ent is ref)
            return np.array([self.asserts(val)])

      class Population(node.Discrete):
         def init(self, config):
            self.max = config.NPOP

      class R(node.Continuous):
         def init(self, config):
            self.min = -config.STIM
            self.max = config.STIM

         def get(self, ent, ref):
            self._val = ent.base.r - ref.base.r
            return np.array([self.norm()])
   
      #You made this continuous
      class C(node.Continuous):
         def init(self, config):
            self.min = -config.STIM
            self.max = config.STIM

         def get(self, ent, ref):
            self._val = ent.base.c - ref.base.c
            return np.array([self.norm()])

      #Historical stats
      class Damage(node.Continuous):
         def init(self, config):
            #This scale may eventually be too high
            self.val   = 0
            self.scale = 0.01

      class TimeAlive(node.Continuous):
         def init(self, config):
            self.val = 0

      #Resources
      class Food(node.Continuous):
         def init(self, config):
            self.val = config.RESOURCE
            self.max = config.RESOURCE

      class Water(node.Continuous):
         def init(self, config):
            self.val = config.RESOURCE
            self.max = config.RESOURCE

      class Health(node.Continuous):
         def init(self, config):
            self.val = config.HEALTH 
            self.max = config.HEALTH

      #Status effects
      class Freeze(node.Continuous):
         def init(self, config):
            self.val = 0
            self.max = 3

      class Immune(node.Continuous):
         def init(self, config):
            self.val = config.IMMUNE
            self.max = config.IMMUNE

      class Wilderness(node.Continuous):
         def init(self, config):
            #You set a low max here
            self.val = -1 
            self.min = -1
            self.max = 99

   class Tile(Config):
      @staticmethod
      def N(config):
         return config.WINDOW**2

      class NEnts(node.Continuous):
         def init(self, config):
            self.max = config.NENT
            self.val = 0

      class Index(node.Discrete):
         def init(self, config):
            self.max = config.NTILE

      class R(node.Discrete):
         def init(self, config):
            self.max = config.TERRAIN_SIZE
 
      class C(node.Discrete):
         def init(self, config):
            self.max = config.TERRAIN_SIZE
