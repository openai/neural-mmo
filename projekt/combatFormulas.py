'''Just a test file for designing combat formulas'''

from pdb import set_trace as T

import numpy as np
from matplotlib import pyplot as plt

#Melee max from 5 to 50
def meleeMax(level):
   return np.floor(5 + level * 45 / 99)

def rangeMax(level):
   return np.floor(3 + level * 32 / 99)

def mageMax(level):
   return np.floor(1 + level * 24 / 99)

def accuracy(defLevel, targDef):
   return 0.5 + (defLevel - targDef) / 200

if __name__ == '__main__':
   levels = np.arange(1, 100)

   plt.plot(levels, meleeMax(levels), 'r')
   plt.plot(levels, rangeMax(levels), 'g')
   plt.plot(levels, mageMax(levels) , 'b')
   #plt.plot(levels, 100*accuracy(50, levels) , 'k')
   plt.show()
   
   
