from pdb import set_trace as T
import numpy as np

class Stim:
   def __init__(self, max=float('inf'), min=0, val=None, scale=1.0):
      self.max = max
      self.min = min
      self._val = val

      assert max > min
      self.range = max - min + 1
      self.scale = scale

   @property
   def val(self):
      return self._val - self.min

class Discrete(Stim):
   @property
   def oneHot(self):
      ary = np.zeros(self.range)
      ary[self.val] = 1
      return ary

class Continuous(Stim):
   @property
   def norm(self):
      assert self.range != float('inf')
      return self.val / self.range - 0.5

   @property
   def scaled(self):
      return self.scale * self.val
