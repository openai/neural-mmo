from pdb import set_trace as T
import abc

import numpy as np
from forge.blade.systems import experience, ai, combat

from forge.blade.lib.enums import Material

class Skills:
   def __init__(self, config):
      expCalc     = experience.ExperienceCalculator()
      self.skills = set()
      self.config = config
   
      #Combat skills
      self.constitution = Constitution(self.skills, expCalc, config)
      self.melee        = Melee(self.skills, expCalc, config)
      self.range        = Range(self.skills, expCalc, config)
      self.mage         = Mage(self.skills, expCalc, config)
      self.defense      = Defense(self.skills, expCalc, config)

      #Harvesting Skills
      self.fishing      = Fishing(self.skills, expCalc, config)
      self.hunting      = Hunting(self.skills, expCalc, config)
      self.mining       = Mining(self.skills, expCalc, config)

      #Processing Skills
      self.cooking      = Cooking(self.skills, expCalc, config)
      self.smithing     = Smithing(self.skills, expCalc, config)

   def packet(self):
      data = {}
      
      data['constitution'] = self.constitution.packet()
      data['melee']        = self.melee.packet()
      data['range']        = self.range.packet()
      data['mage']         = self.mage.packet()
      data['defense']      = self.defense.packet()
      data['fishing']      = self.fishing.packet()
      data['hunting']      = self.hunting.packet()
      data['cooking']      = self.cooking.packet()
      data['smithing']     = self.smithing.packet()
      data['level']        = combat.level(self)

      return data

   def update(self, ent, world, actions):
      for skill in self.skills:
         skill.update(ent, world)

   def applyDamage(self, dmg, style):
      config = self.config
      scale  = config.XP_SCALE
      self.constitution.exp += scale * dmg * config.CONSTITUTION_XP_SCALE

      skill = self.__dict__[style]
      skill.exp += scale * dmg * config.COMBAT_XP_SCALE

   def receiveDamage(self, dmg):
      scale = self.config.XP_SCALE
      self.constitution.exp += scale * dmg * 2
      self.defense.exp      += scale * dmg * 4

class Skill:
   skillItems = abc.ABCMeta
   def __init__(self, skills, expCalc, config):
      self.expCalc = expCalc
      self.exp     = 0

      self.config  = config
      skills.add(self)

   def packet(self):
      data = {}
      
      data['exp']   = self.exp
      data['level'] = self.level

      return data

   def update(self, ent, world):
      pass

   @property
   def level(self):
      lvl = self.expCalc.levelAtExp(self.exp)
      assert lvl == int(lvl)
      return int(lvl)

class CombatSkill(Skill): pass

class Constitution(CombatSkill):
   def __init__(self, skills, expCalc, config):
      super().__init__(skills, expCalc, config)
      self.exp = self.expCalc.expAtLevel(config.HEALTH)

   def update(self, ent, world):
      health = ent.resources.health
      food   = ent.resources.food
      water  = ent.resources.water

      config = self.config

      #Heal if above fractional resource threshold
      foodThresh  = food.val  > config.HEALTH_REGEN_THRESHOLD * ent.skills.hunting.level
      waterThresh = water.val > config.HEALTH_REGEN_THRESHOLD * ent.skills.fishing.level

      if foodThresh and waterThresh:
         restore = np.floor(self.level * self.config.HEALTH_RESTORE)
         restore = min(self.level - health.val, restore)
         health.val += restore

      if food.val <= 0:
         health.val -= 1

      if water.val <= 0:
         health.val -= 1

class Melee(CombatSkill): pass
class Range(CombatSkill): pass
class Mage(CombatSkill): pass
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

class Fishing(HarvestingSkill):
   def __init__(self, skills, expCalc, config):
      super().__init__(skills, expCalc, config)
      self.exp = self.expCalc.expAtLevel(config.RESOURCE)

   def update(self, ent, world):
      water      = ent.resources.water
      water.val -= 1

      if Material.WATER.value not in ai.adjacentMats(world.env, ent.base.pos):
         return

      restore    = np.floor(self.level * self.config.RESOURCE_RESTORE)
      restore    = min(self.level - water.val, restore)
      water.val += restore

      scale = self.config.XP_SCALE
      self.exp += scale * restore;

class Hunting(HarvestingSkill):
   def __init__(self, skills, expCalc, config):
      super().__init__(skills, expCalc, config)
      self.exp = self.expCalc.expAtLevel(config.RESOURCE)

   def update(self, ent, world):
      food      = ent.resources.food
      food.val -= 1

      r, c = ent.base.pos
      if (type(world.env.tiles[r, c].mat) not in [Material.FOREST.value] or 
            not world.env.harvest(r, c)):
         return

      restore   = np.floor(self.level * self.config.RESOURCE_RESTORE)
      restore   = min(self.level - food.val, restore)
      food.val += restore

      scale = self.config.XP_SCALE
      self.exp += scale * restore;

class Mining(HarvestingSkill): pass

class ProcessingSkill(NonCombatSkill):
   def process(self, inv, item):
      self.attempt(inv, item)
     
class Cooking(ProcessingSkill): pass
class Smithing(ProcessingSkill): pass        


