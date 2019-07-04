import abc

import numpy as np
from forge.blade.systems import experience

class Skills:
   def __init__(self, config):
      expCalc = experience.ExperienceCalculator()
   
      #Combat skills
      self.constitution = Constitution(expCalc)
      self.melee        = Melee(expCalc)
      self.ranged       = Ranged(expCalc)
      self.defense      = Defense(expCalc)

      #Harvesting Skills
      self.fishing      = Fishing(expCalc)
      self.mining       = Mining(expCalc)

      #Processing Skills
      self.cooking      = Cooking(expCalc)
      self.smithing     = Smithing(expCalc)

   def packet(self):
      data = {}
      
      data['constitution'] = self.constitution.packet()
      data['melee']        = self.melee.packet()
      data['ranged']       = self.ranged.packet()
      data['defense']      = self.defense.packet()
      data['fishing']      = self.fishing.packet()
      data['mining']       = self.mining.packet()
      data['cooking']      = self.cooking.packet()
      data['smithing']     = self.smithing.packet()

   def update(self, world, actions):
      return 

class Skill:
   skillItems = abc.ABCMeta
   def __init__(self, expCalc):
      self.expCalc = expCalc
      self.exp     = 0

   def packet(self):
      data = {}
      
      data['exp']   = self.exp
      data['level'] = self.level

   @property
   def level(self):
      return self.expCalc.levelAtExp(self.exp)

class CombatSkill(Skill):
   def addCombatExp(self, skill, dmg):
      skill.exp             += int(3*dmg)
      self.constitution.exp += int(dmg)

class Constitution(CombatSkill):
   def __init__(self, expCalc):
      super().__init__(expCalc)
      self.exp = self.expCalc.expAtLevel(10)

class Melee(CombatSkill): pass
class Ranged(CombatSkill): pass
class Defense(CombatSkill): pass

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

class Fishing(HarvestingSkill): pass
class Mining(HarvestingSkill): pass


class ProcessingSkill(NonCombatSkill):
   def process(self, inv, item):
      self.attempt(inv, item)
     
class Cooking(ProcessingSkill): pass
class Smithing(ProcessingSkill): pass        


