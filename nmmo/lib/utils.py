import itertools
import time

import numpy as np
from collections import defaultdict

class staticproperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)

def seed():
   return int(np.random.randint(0, 2**32))

def linf(pos1, pos2):
   r1, c1 = pos1
   r2, c2 = pos2
   return max(abs(r1 - r2), abs(c1 - c2))

#Bounds checker
def inBounds(r, c, shape, border=0):
   R, C = shape
   return (
         r > border and
         c > border and
         r < R - border and
         c < C - border
         )
