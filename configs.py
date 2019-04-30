from pdb import set_trace as T
from forge.blade.core.config import Config
from forge.blade.lib import utils
import numpy as np

class Experiment(Config):
   MODELDIR='resource/logs/'
   EMBED   = 128
   HIDDEN  = 128
   NHEAD   = 8
   NROLLOUTS = 1
   UPDATEFREQ = 100 / NROLLOUTS
   TEST = False
   LOAD = False
   BEST = False
   SAMPLE = False
   NATTN = 2
   NPOP = 1
   SHAREINIT = False
   ENTROPY = 0.01
   VAMPYR = 1
   AUTO_TARGET = False

#Foraging only
class Law(Experiment):
   pass

#Foraging + Combat
class Chaos(Experiment):
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
