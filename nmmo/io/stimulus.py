from pdb import set_trace as T
import numpy as np

from nmmo.lib import utils

class SerializedVariable:
   CONTINUOUS = False
   DISCRETE   = False
   def __init__(self, dataframe, key, val=None, config=None):
      if config is None:
         config    = dataframe.config
 
      self.obj  = str(self.__class__).split('.')[-2]
      self.attr = self.__class__.__name__
      self.key  = key

      self.min = 0
      self.max = np.inf
      self.val = val

      self.dataframe = dataframe
      self.init(config)
      err = 'Must set a default val upon instantiation or init()'
      assert self.val is not None, err

      #Update dataframe
      if dataframe is not None:
         self.update(self.val)

   #Defined for cleaner stim files
   def init(self):
      pass

   def packet(self):
      return {
            'val': self.val,
            'max': self.max}

   def update(self, val):
      self.val = min(max(val, self.min), self.max)
      self.dataframe.update(self, self.val)
      return self

   def increment(self, val=1):
      self.update(self.val + val)
      return self

   def decrement(self, val=1):
      self.update(self.val - val)
      return self

   @property
   def empty(self):
      return self.val == 0

   def __add__(self, other):
      self.increment(other)
      return self

   def __sub__(self, other):
      self.decrement(other)
      return self

   def __eq__(self, other):
      return self.val == other

   def __ne__(self, other):
      return self.val != other

   def __lt__(self, other):
      return self.val < other

   def __le__(self, other):
      return self.val <= other

   def __gt__(self, other):
      return self.val > other

   def __ge__(self, other):
      return self.val >= other

class Continuous(SerializedVariable):
   CONTINUOUS = True

class Discrete(Continuous):
   DISCRETE = True


class Serialized(metaclass=utils.IterableNameComparable):
   def dict():
      return {k[0] : v for k, v in dict(Stimulus).items()}

   class Entity(metaclass=utils.IterableNameComparable):
      @staticmethod
      def N(config):
         return config.N_AGENT_OBS

      class Self(Discrete):
         def init(self, config):
            self.max = 1
            self.scale = 1.0

      class ID(Continuous):
         def init(self, config):
            self.min   = -np.inf
            self.scale = 0.001

      class AttackerID(Continuous):
         def init(self, config):
            self.min   = -np.inf
            self.scale = 0.001

      class Level(Continuous):
         def init(self, config):
            self.scale = 0.05

      class Population(Discrete):
         def init(self, config):
            self.min = -3 #NPC index
            self.max = config.NPOP - 1
            self.scale = 1.0

      class R(Discrete):
         def init(self, config):
            self.min = 0
            self.max = config.TERRAIN_SIZE - 1
            self.scale = 0.15

      class C(Discrete):
         def init(self, config):
            self.min = 0
            self.max = config.TERRAIN_SIZE - 1
            self.scale = 0.15

      #Historical stats
      class Damage(Continuous):
         def init(self, config):
            #This scale may eventually be too high
            self.val   = 0
            self.scale = 0.1

      class TimeAlive(Continuous):
         def init(self, config):
            self.val = 0
            self.scale = 0.01

      #Resources -- Redo the max/min scaling. You can't change these
      #after init without messing up the embeddings
      class Food(Continuous):
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

      class Water(Continuous):
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

      class Health(Continuous):
         def init(self, config):
            self.val = config.BASE_HEALTH
            self.max = config.BASE_HEALTH
            self.scale = 0.1

      #Status effects
      class Freeze(Continuous):
         def init(self, config):
            self.val = 0
            self.max = 3
            self.scale = 0.3

   class Tile(metaclass=utils.IterableNameComparable):
      @staticmethod
      def N(config):
         return config.WINDOW**2

      class NEnts(Continuous):
         def init(self, config):
            self.max = config.NENT
            self.val = 0
            self.scale = 1.0

      class Index(Discrete):
         def init(self, config):
            self.max = config.NTILE
            self.scale = 0.15

      class R(Discrete):
         def init(self, config):
            self.max = config.TERRAIN_SIZE - 1
            self.scale = 0.15
 
      class C(Discrete):
         def init(self, config):
            self.max = config.TERRAIN_SIZE - 1
            self.scale = 0.15

for objName, obj in Serialized:
   for idx, (attrName, attr) in enumerate(obj):
      attr.index = idx 


