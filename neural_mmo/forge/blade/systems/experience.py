from pdb import set_trace as T
import numpy as np

class ExperienceCalculator:
   def __init__(self):
      self.exp = [0]
      self.tabulateExp()

   def tabulateExp(self, numLevels=99):
      for i in range(2, numLevels+1):
         increment = np.floor(i-1 + 300*(2**((i-1)/7.0)))/4.0
         self.exp += [self.exp[-1] + increment]

      self.exp = np.floor(np.array(self.exp))

   def expAtLevel(self, level):
      return self.exp[level - 1]

   def levelAtExp(self, exp):
      if exp >= self.exp[-1]:
         return len(self.exp)
      return np.argmin(exp >= self.exp)
