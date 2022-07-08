import numpy as np

class Empty():
    def roll(self, realm, level):
        return []

class Fixed():
    def __init__(self, item, amount=1):
        self.item = item
        self.amount = amount

    def roll(self, realm, level):
        return [self.item(realm, level, amount=amount)]

class Drop:
   def __init__(self, item, amount, prob):
      self.item = item
      self.amount = amount
      self.prob = prob

   def roll(self, realm, level):
      if np.random.rand() < self.prob:
         return self.item(realm, level, quantity=self.amount)

class Standard:
   def __init__(self):
      self.drops = []

   def add(self, item, quant=1, prob=1.0):
      self.drops += [Drop(item, quant, prob)]

   def roll(self, realm, level):
      ret = []
      for e in self.drops:
         drop = e.roll(realm, level)
         if drop is not None:
            ret += [drop]
      return ret

class Empty(Standard):
    def roll(self, realm, level):
        return []

class Ammunition(Standard):
    def __init__(self, item):
        self.item = item

    def roll(self, realm, level):
        return [self.item(realm, level)]

class Consumable(Standard):
    def __init__(self, item):
        self.item = item

    def roll(self, realm, level):
        return [self.item(realm, level)]
