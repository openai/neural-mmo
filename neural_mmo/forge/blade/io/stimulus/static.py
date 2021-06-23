from pdb import set_trace as T
import numpy as np

from neural_mmo.forge.blade.io import node

def bind(gameCls):
   def to(ioCls):
      @property
      def GAME_CLS():
         return gameCls

      ioCls.GAME_CLS = GAME_CLS
      #ioCls.GAME_CLS = gameCls
      return ioCls
   return to 

class Config(metaclass=node.IterableNameComparable):
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
            self.max = 1
            self.scale = 1.0

      class ID(node.Continuous):
         def init(self, config):
            self.min   = -np.inf
            self.scale = 0.001

      class AttackerID(node.Continuous):
         def init(self, config):
            self.min   = -np.inf
            self.scale = 0.001

      class Level(node.Continuous):
         def init(self, config):
            self.scale = 0.05

      class Population(node.Discrete):
         def init(self, config):
            self.min = -3 #NPC index
            self.max = config.NPOP - 1
            self.scale = 1.0

      class R(node.Discrete):
         def init(self, config):
            self.min = 0
            self.max = config.TERRAIN_SIZE - 1
            self.scale = 0.15

      class C(node.Discrete):
         def init(self, config):
            self.min = 0
            self.max = config.TERRAIN_SIZE - 1
            self.scale = 0.15

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
            if config.game_system_enabled('Progression'):
               self.val = config.PROGRESSION_BASE_RESOURCE
               self.max = config.PROGRESSION_BASE_RESOURCE
            elif config.game_system_enabled('Resource'):
               self.val = config.RESOURCE_BASE_RESOURCE
               self.max = config.RESOURCE_BASE_RESOURCE
            else:
               self.val = 1
               self.max = 1
    
            self.scale = 0.1

      class Water(node.Continuous):
         def init(self, config):
            if config.game_system_enabled('Progression'):
               self.val = config.PROGRESSION_BASE_RESOURCE
               self.max = config.PROGRESSION_BASE_RESOURCE
            elif config.game_system_enabled('Resource'):
               self.val = config.RESOURCE_BASE_RESOURCE
               self.max = config.RESOURCE_BASE_RESOURCE
            else:
               self.val = 1
               self.max = 1
 
            self.scale = 0.1

      class Health(node.Continuous):
         def init(self, config):
            self.val = config.BASE_HEALTH
            self.max = config.BASE_HEALTH
            self.scale = 0.1

      #Status effects
      class Freeze(node.Continuous):
         def init(self, config):
            self.val = 0
            self.max = 3
            self.scale = 0.3

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
            self.max = config.TERRAIN_SIZE - 1
            self.scale = 0.15
 
      class C(node.Discrete):
         def init(self, config):
            self.max = config.TERRAIN_SIZE - 1
            self.scale = 0.15

for objName, obj in Stimulus:
   for idx, (attrName, attr) in enumerate(obj):
      attr.index = idx 
