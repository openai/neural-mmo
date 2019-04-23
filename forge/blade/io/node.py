from pdb import set_trace as T
import numpy as np

from forge.blade.lib.utils import staticproperty, classproperty

class Stim:
   max = float('inf')
   min = 0

   default = 0
   def __init__(self):
      cls = self.__class__
      assert cls.max > cls.min
      self._val = cls.default

   @classproperty
   def name(self):
      name = self.__name__
      return name[0].lower() + name[1:]

   @property
   def val(self):
      if self._val is None:
         return 0
      return self._val

   def update(self, val):
      if self._val is not None:
         assert val >= self.min and val <= self.max
      self._val = val
      return self #for convenience

   def packet(self):
      return {'val': self.val, 'max': self.max}

   def get(self, *args):
      return self.val

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
   @classproperty
   def range(cls):
      return cls.max - cls.min + 1

   def oneHot(self):
      ary = np.zeros(self.range)
      ary[self.norm()] = 1
      return ary

   def norm(self):
      val = self.val - self.min
      assert val == int(val)
      return int(val)

class Continuous(Stim):
   @classproperty
   def range(cls):
      return cls.max - cls.min

   def norm(self, val):
      assert val >= self.min and val <= self.max
      if self.range == float('inf'):
         return self.scaled(val)
      return val / self.range - 0.5

   def scaled(self, val):
      return self.scale * val



