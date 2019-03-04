from pdb import set_trace as T
from forge.blade.core.config import Config
from forge.blade.lib import utils
import numpy as np

class Experiment(Config):
   def defaults(self):
      super().defaults()
      self.MODELDIR='resource/logs/'
      self.HIDDEN = 32
      self.TEST = False
      self.LOAD = False
      self.BEST = False
      self.SAMPLE = False
      self.NATTN = 2
      self.NPOP = 1
      self.SHAREINIT = False
      self.ENTROPY = 0.01
      self.VAMPYR = 1
      self.AUTO_TARGET = False

#Foraging only
class Law(Experiment):
   def defaults(self):
      super().defaults()

   #Damage 
   def MELEEDAMAGE(self, ent, targ): return 0
   def RANGEDAMAGE(self, ent, targ): return 0
   def MAGEDAMAGE(self, ent, targ): return 0

#Foraging + Combat
class Chaos(Experiment):
   def defaults(self):
      super().defaults()
      self.RANGERANGE = 2
      self.MAGERANGE  = 3

   def vamp(self, ent, targ, frac, dmg):
      dmg = int(frac * dmg)
      targ.food.decrement(amt=dmg)
      targ.water.decrement(amt=dmg)
      ent.food.increment(amt=dmg)
      ent.water.increment(amt=dmg)

   #Damage formulas. Lambdas don't pickle well
   def MELEEDAMAGE(self, ent, targ):
      dmg = 10
      targ.applyDamage(dmg)
      self.vamp(ent, targ, self.VAMPYR, dmg)
      return dmg

   def RANGEDAMAGE(self, ent, targ):
      dmg = 2
      targ.applyDamage(dmg)
      self.vamp(ent, targ, self.VAMPYR, dmg)
      return dmg

   def MAGEDAMAGE(self, ent, targ):
      dmg = 1
      targ.applyDamage(dmg)
      self.vamp(ent, targ, self.VAMPYR, dmg)
      return dmg
