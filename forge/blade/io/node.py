from pdb import set_trace as T
import numpy as np
import gym

import inspect
from enum import Enum, auto

from forge.blade.lib.utils import classproperty, staticproperty

class IterableTypeCompare(type):
   def __iter__(cls):
      stack = list(cls.__dict__.items())
      while len(stack) > 0:
         name, attr = stack.pop()
         if type(name) != tuple:
            name = tuple([name])
         if not inspect.isclass(attr):
            continue
         if issubclass(attr, Flat):
            for n, a in attr.__dict__.items():
               n = name + tuple([n])
               stack.append((n, a))
            continue
         yield name, attr

   def values(cls):
      return [e[1] for e in cls]

   def __hash__(self):
      return hash(self.__name__)

   def __eq__(self, other):
      return self.__name__ == other.__name__

   def __ne__(self, other):
      return self.__name__ != other.__name__

   def __lt__(self, other):
      return self.__name__ < other.__name__

   def __le__(self, other):
      return self.__name__ <= other.__name__

   def __gt__(self, other):
      return self.__name__ > other.__name__

   def __ge__(self, other):
      return self.__name__ >= other.__name__

class Flat:
   pass

class Stim(metaclass=IterableTypeCompare):
   default = 0
   max = np.inf
   min = 0

   def __init__(self, config):
      cls = self.__class__
      self.min = cls.min
      self.max = cls.max
      self.default = cls.default

      self.init(config)
      self._val = self.default

   def init(self, config):
      pass

   @classproperty
   def name(self):
      name = self.__name__
      return name[0].lower() + name[1:]

   @property
   def val(self):
      if self._val is None:
         return 0
      return self._val

   def asserts(self, val):
      if val is not None:
         assert val >= self.min and val <= self.max, str(self) + ': ' + str(val)
      return val

   def update(self, val):
      self._val = val
      return self #for convenience

   def packet(self):
      return {
            'val': self.val,
            'max': self.max if self.max != float('inf') else None}

   def get(self, *args):
      return self.asserts(self.val)

   @property
   def missing(self):
      return self.max - self.val

   def increment(self, amt=1):
      self._val = min(self.max, self.val + amt)

   def decrement(self, amt=1):
      self._val = max(0, self.val - amt)

   def __add__(self, other):
      self.increment(other)
      return self

   def __sub__(self, other):
      self.decrement(other)
      return self

   def __lt__(self, other):
      return self.val < other

   def __le__(self, other):
      return self.val <= other

   def __gt__(self, other):
      return self.val > other

   def __ge__(self, other):
      return self.val >= other

class Discrete(Stim):
   def __init__(self, config):
      super().__init__(config)
      self.space = gym.spaces.Box(
            low=np.float32(0.0),
            high=np.float32(self.range),
            shape=(1,))

   @property
   def range(self):
      return self.max - self.min + 1

   def oneHot(self):
      ary = np.zeros(self.range)
      ary[self.norm()] = 1
      return ary

   def norm(self):
      val = self.val# - self.min
      assert val == int(val)
      return int(val)

   def get(self, *args):
      self.asserts(self.val)
      return np.array([self.norm()])
      return self.norm()

      #No norm needed for discrete vars. Below is for
      #current hack where RLlib treats everything as continuous
      #The default preprocessor won't norm, so we can still embed
      return self.norm()

class Continuous(Stim):
   def __init__(self, config):
      super().__init__(config)
      self.space = gym.spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(1,))

   @property
   def range(self):
      return self.max - self.min

   def norm(self):
      assert self.val >= self.min and self.val <= self.max
      val = self.val - self.min
      if self.range == np.inf:
         val = self.scaled(val)
      else:
         val = 2*(val / self.range) - 1
      assert val >= -1 and val <= 1, self
      return val

   def scaled(self, val):
      return self.scale * val

   def get(self, *args):
      self.asserts(self.val)
      val = self.norm()
      return np.array([val])

class NodeType(Enum):
   #Tree edges
   STATIC = auto()    #Traverses all edges without decisions
   SELECTION = auto() #Picks an edge to follow

   #Executable actions
   ACTION    = auto() #No arguments
   CONSTANT  = auto() #Constant argument
   VARIABLE  = auto() #Variable argument

class Node(metaclass=IterableTypeCompare):
   SERIAL = 2

   @staticproperty
   def edges():
      return []

   #Fill these in
   @staticproperty
   def priority():
      return None

   @staticproperty
   def type():
      return None

   @staticproperty
   def leaf():
      return False

   @classmethod
   def N(cls, config):
      return len(cls.edges)

   def args(stim, entity, config):
      return []
