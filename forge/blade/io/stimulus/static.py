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
         #return config.WINDOW ** 2
         return config.N_AGENT_OBS

      class Self(node.Discrete):
         def init(self, config):
            self.val = 0
            self.max = 1
            self.scale = 1.0

         def get(self, ent, ref):
            val = int(ent is ref)
            return np.array([self.asserts(val)])

      class Population(node.Discrete):
         def init(self, config):
            self.max = config.NPOP
            self.scale = 1.0

      class R(node.Discrete):
         def init(self, config):
            #self.min = -config.STIM
            #self.max = config.STIM
            self.min = 0
            self.max = config.TERRAIN_SIZE
            self.scale = 0.15

         def get(self, ent, ref):
            self._val = ent.base.r - ref.base.r
            return np.array([self.norm()])
   
      #You made this continuous
      class C(node.Discrete):
         def init(self, config):
            #self.min = -config.STIM
            #self.max = config.STIM
            self.min = 0
            self.max = config.TERRAIN_SIZE
            self.scale = 0.15


         def get(self, ent, ref):
            self._val = ent.base.c - ref.base.c
            return np.array([self.norm()])

      #Historical stats
      class Damage(node.Continuous):
         def init(self, config):
            #This scale may eventually be too high
            self.val   = 0
            self.scale = 0.1

      class TimeAlive(node.Continuous):
         def init(self, config):
            self.val = 0
            self.scale = 0.01

      #Resources -- Redo the max/min scaling. You can't change these
      #after init without messing up the embeddings
      class Food(node.Continuous):
         def init(self, config):
            self.val = config.RESOURCE
            self.max = config.RESOURCE
            self.scale = 0.1

      class Water(node.Continuous):
         def init(self, config):
            self.val = config.RESOURCE
            self.max = config.RESOURCE
            self.scale = 0.1

      class Health(node.Continuous):
         def init(self, config):
            self.val = config.HEALTH 
            self.max = config.HEALTH
            self.scale = 0.1

      #Status effects
      class Freeze(node.Continuous):
         def init(self, config):
            self.val = 0
            self.max = 3
            self.scale = 0.3

      class Immune(node.Continuous):
         def init(self, config):
            self.val = config.IMMUNE
            self.max = config.IMMUNE
            self.scale = 0.1

      class Wilderness(node.Continuous):
         def init(self, config):
            #You set a low max here
            self.val = -1 
            self.min = -1
            self.max = 99
            self.scale = 0.01

   class Tile(Config):
      @staticmethod
      def N(config):
         return config.WINDOW**2

      class NEnts(node.Continuous):
         def init(self, config):
            self.max = config.NENT
            self.val = 0
            self.scale = 1.0

      class Index(node.Discrete):
         def init(self, config):
            self.max = config.NTILE
            self.scale = 0.15

      class R(node.Discrete):
         def init(self, config):
            self.max = config.TERRAIN_SIZE
            self.scale = 0.15
 
      class C(node.Discrete):
         def init(self, config):
            self.max = config.TERRAIN_SIZE
            self.scale = 0.15
