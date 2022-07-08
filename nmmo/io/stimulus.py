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
      def enabled(config):
         return True

      @staticmethod
      def N(config):
         return config.PLAYER_N_OBS

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

      class ItemLevel(Continuous):
         def init(self, config):
            self.scale = 0.025
            self.max   = 5 * config.NPC_LEVEL_MAX

      class Comm(Discrete):
         def init(self, config):
            self.scale = 0.025
            self.max = 1
            if config.COMMUNICATION_SYSTEM_ENABLED:
                self.max   = config.COMMUNICATION_NUM_TOKENS

      class Population(Discrete):
         def init(self, config):
            self.min = -3 #NPC index
            self.max = config.PLAYER_POLICIES - 1
            self.scale = 1.0

      class R(Discrete):
         def init(self, config):
            self.min = 0
            self.max = config.MAP_SIZE - 1
            self.scale = 0.15

      class C(Discrete):
         def init(self, config):
            self.min = 0
            self.max = config.MAP_SIZE - 1
            self.scale = 0.15

      # Historical stats
      class Damage(Continuous):
         def init(self, config):
            #This scale may eventually be too high
            self.val   = 0
            self.scale = 0.1

      class TimeAlive(Continuous):
         def init(self, config):
            self.val = 0
            self.scale = 0.01

      # Status effects
      class Freeze(Continuous):
         def init(self, config):
            self.val = 0
            self.max = 3
            self.scale = 0.3

      class Gold(Continuous):
         def init(self, config):
            self.val = 0
            self.scale = 0.01

      # Resources -- Redo the max/min scaling. You can't change these
      # after init without messing up the embeddings
      class Health(Continuous):
         def init(self, config):
            self.val = config.PLAYER_BASE_HEALTH
            self.max = config.PLAYER_BASE_HEALTH
            self.scale = 0.1

      class Food(Continuous):
         def init(self, config):
            if config.RESOURCE_SYSTEM_ENABLED:
               self.val = config.RESOURCE_BASE
               self.max = config.RESOURCE_BASE
            else:
               self.val = 1
               self.max = 1
    
            self.scale = 0.01

      class Water(Continuous):
         def init(self, config):
            if config.RESOURCE_SYSTEM_ENABLED:
               self.val = config.RESOURCE_BASE
               self.max = config.RESOURCE_BASE
            else:
               self.val = 1
               self.max = 1
 
            self.scale = 0.01

      class Melee(Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.PROGRESSION_SYSTEM_ENABLED:
                self.max = config.PROGRESSION_LEVEL_MAX

      class Range(Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.PROGRESSION_SYSTEM_ENABLED:
                self.max = config.PROGRESSION_LEVEL_MAX

      class Mage(Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.PROGRESSION_SYSTEM_ENABLED:
                self.max = config.PROGRESSION_LEVEL_MAX

      class Fishing(Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.PROGRESSION_SYSTEM_ENABLED:
                self.max = config.PROGRESSION_LEVEL_MAX

      class Herbalism(Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.PROGRESSION_SYSTEM_ENABLED:
                self.max = config.PROGRESSION_LEVEL_MAX

      class Prospecting(Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.PROGRESSION_SYSTEM_ENABLED:
                self.max = config.PROGRESSION_LEVEL_MAX

      class Carving(Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.PROGRESSION_SYSTEM_ENABLED:
                self.max = config.PROGRESSION_LEVEL_MAX

      class Alchemy(Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.PROGRESSION_SYSTEM_ENABLED:
                self.max = config.PROGRESSION_LEVEL_MAX

   class Tile(metaclass=utils.IterableNameComparable):
      @staticmethod
      def enabled(config):
         return True

      @staticmethod
      def N(config):
         return config.MAP_N_OBS

      class NEnts(Continuous):
         def init(self, config):
            self.max = config.PLAYER_N
            self.val = 0
            self.scale = 1.0

      class Index(Discrete):
         def init(self, config):
            self.max = config.MAP_N_TILE
            self.scale = 0.15

      class R(Discrete):
         def init(self, config):
            self.max = config.MAP_SIZE - 1
            self.scale = 0.15
 
      class C(Discrete):
         def init(self, config):
            self.max = config.MAP_SIZE - 1
            self.scale = 0.15

   class Item(metaclass=utils.IterableNameComparable):
      @staticmethod
      def enabled(config):
         return config.ITEM_SYSTEM_ENABLED

      @staticmethod
      def N(config):
         return config.ITEM_N_OBS

      class ID(Continuous):
         def init(self, config):
            self.scale = 0.001

      class Index(Discrete):
         def init(self, config):
            self.max   = config.ITEM_N + 1
            self.scale = 1.0 / self.max

      class Level(Continuous):
         def init(self, config):
            self.max   = 99
            self.scale = 1.0 / self.max

      class Capacity(Continuous):
         def init(self, config):
            self.max   = 99
            self.scale = 1.0 / self.max

      class Quantity(Continuous):
         def init(self, config):
            self.max   = 99
            self.scale = 1.0 / self.max

      class Tradable(Discrete):
         def init(self, config):
            self.max   = 1
            self.scale = 1.0

      class MeleeAttack(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class RangeAttack(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class MageAttack(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class MeleeDefense(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class RangeDefense(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class MageDefense(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class HealthRestore(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class ResourceRestore(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class Price(Continuous):
         def init(self, config):
            self.scale = 0.01

      class Equipped(Discrete):
         def init(self, config):
            self.scale = 1.0

   # TODO: Figure out how to autogen this from Items
   class Market(metaclass=utils.IterableNameComparable):
      @staticmethod
      def enabled(config):
         return config.EXCHANGE_SYSTEM_ENABLED

      @staticmethod
      def N(config):
         return config.EXCHANGE_N_OBS

      class ID(Continuous):
         def init(self, config):
            self.scale = 0.001

      class Index(Discrete):
         def init(self, config):
            self.max   = config.ITEM_N + 1
            self.scale = 1.0 / self.max

      class Level(Continuous):
         def init(self, config):
            self.max   = 99
            self.scale = 1.0 / self.max

      class Capacity(Continuous):
         def init(self, config):
            self.max   = 99
            self.scale = 1.0 / self.max

      class Quantity(Continuous):
         def init(self, config):
            self.max   = 99
            self.scale = 1.0 / self.max

      class Tradable(Discrete):
         def init(self, config):
            self.max   = 1
            self.scale = 1.0

      class MeleeAttack(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class RangeAttack(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class MageAttack(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class MeleeDefense(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class RangeDefense(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class MageDefense(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class HealthRestore(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class ResourceRestore(Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class Price(Continuous):
         def init(self, config):
            self.scale = 0.01

      class Equipped(Discrete):
         def init(self, config):
            self.scale = 1.0


for objName, obj in Serialized:
   for idx, (attrName, attr) in enumerate(obj):
      attr.index = idx 
