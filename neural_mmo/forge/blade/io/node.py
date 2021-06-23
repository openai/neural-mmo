from pdb import set_trace as T
import numpy as np
import gym

import inspect
from enum import Enum, auto
from collections import deque

from neural_mmo.forge.blade.lib.utils import classproperty, staticproperty

class Iterable(type):
   def __iter__(cls):
      queue = deque(cls.__dict__.items())
      while len(queue) > 0:
         name, attr = queue.popleft()
         if type(name) != tuple:
            name = tuple([name])
         if not inspect.isclass(attr):
            continue
         if issubclass(attr, Flat):
            for n, a in attr.__dict__.items():
               n = name + tuple([n])
               queue.append((n, a))
            continue
         yield name, attr

   def values(cls):
      return [e[1] for e in cls]

class NameComparable(type):
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

class IterableNameComparable(Iterable, NameComparable):
   pass

class Flat:
   pass

class Stim:
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

class Continuous(Stim):
   CONTINUOUS = True

class Discrete(Continuous):
   DISCRETE = True

class NodeType(Enum):
   #Tree edges
   STATIC = auto()    #Traverses all edges without decisions
   SELECTION = auto() #Picks an edge to follow

   #Executable actions
   ACTION    = auto() #No arguments
   CONSTANT  = auto() #Constant argument
   VARIABLE  = auto() #Variable argument

class Node(metaclass=IterableNameComparable):
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
