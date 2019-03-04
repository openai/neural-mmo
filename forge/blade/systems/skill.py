import abc

import numpy as np
from forge.blade.systems import experience

class Skills:
   def __init__(self):
      self.melee        = Melee()
      self.ranged       = Ranged()
      self.defense      = Defense()
      self.constitution = Constitution()

      self.fishing = Fishing()
      self.mining = Mining()

      self.cooking = Cooking()
      self.smithing = Smithing()

   def addCombatExp(self, skill, dmg):
      skill.exp += int(3*dmg)
      self.constitution.exp += int(dmg)

class Skill:
   skillItems = abc.ABCMeta
   expCalculator = experience.ExperienceCalculator()
   def __init__(self):
      self.exp = 0

   @property
   def level(self):
      return self.expCalculator.levelAtExp(self.exp)

class CombatSkill(Skill):
   @property
   def isMelee(self):
      return False

   @property
   def isRanged(self):
      return False

class NonCombatSkill(Skill):
   def success(self, levelReq):
      level = self.level
      if level < levelReq:
         return False
      chance = 0.5 + 0.05*(level - levelReq)
      if chance >= 1.0:
         return True
      return np.random.rand() < chance

   def attempt(self, inv, item):
      if (item.createSkill != self.__class__ or
            self.level < item.createLevel):
         return      

      if item.recipe is not None:
         #Check that everything is available
         if not inv.satisfies(item.recipe): return
         inv.removeRecipe(item.recipe)

      if item.alwaysSucceeds or self.success(item.createLevel):
         inv.add(item, item.amtMade)
         self.exp += item.exp
         return True

class HarvestingSkill(NonCombatSkill):
   #Attempt each item from highest to lowest tier until success
   def harvest(self, inv):
      for e in self.skillItems:
         if self.attempt(inv, e):
            return

class ProcessingSkill(NonCombatSkill):
   def process(self, inv, item):
      self.attempt(inv, item)
     
class Fishing(HarvestingSkill): pass
   
class Mining(HarvestingSkill): pass

class Cooking(ProcessingSkill): pass

class Smithing(ProcessingSkill): pass        

class Melee(CombatSkill):
   @property
   def isMelee(self):
      return True

class Ranged(CombatSkill):
   @property
   def isRanged(self):
      return True

class Defense(CombatSkill): pass

class Constitution(CombatSkill): pass

