import numpy as np

class ExperienceCalculator:
   def __init__(self):
      self.exp = []
      self.tabulateExp()

   def tabulateExp(self, numLevels=99):
      for i in range(numLevels):
         if i == 0:
            self.exp += [0]
         else:
            increment = np.floor(np.floor(i + 300*(2**(i/7.0)))/4.0)
            self.exp += [self.exp[-1] + increment]

   def expAtLevel(self, level):
      return self.exp[level]

   def levelAtExp(self, exp):
      return np.searchsorted(self.exp, exp)
