from pdb import set_trace as T
import numpy as np

from forge.blade.lib.utils import classproperty

class Flat:
   pass

class Stim:
   default = 0
   max = float('inf')
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
         assert val >= self.min and val <= self.max
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
   @property
   def range(self):
      return self.max - self.min + 1

   def oneHot(self):
      ary = np.zeros(self.range)
      ary[self.norm()] = 1
      return ary

   def norm(self):
      val = self.val - self.min
      assert val == int(val)
      return int(val)

   def get(self, *args):
      self.asserts(self.val)
      return self.norm()

class Continuous(Stim):
   @property
   def range(self):
      return self.max - self.min

   def norm(self):
      assert self.val >= self.min and self.val <= self.max
      val = self.val - self.min
      if self.range == float('inf'):
         return self.scaled(val)
      return val / self.range - 0.5

   def scaled(self, val):
      return self.scale * val

   def get(self, *args):
      self.asserts(self.val)
      return self.norm()


