from pdb import set_trace as T
import numpy as np

class ExperienceCalculator:
   def __init__(self, num_levels=15):
      self.exp = np.array([0] + [10*2**i for i in range(num_levels)])

   def expAtLevel(self, level):
      return int(self.exp[level - 1])

   def levelAtExp(self, exp):
      if exp >= self.exp[-1]:
         return len(self.exp)
      return np.argmin(exp >= self.exp)
