from pdb import set_trace as T
import abc

import numpy as np
from forge.blade.systems import experience, combat, ai

from forge.blade.lib import material

### Infrastructure ###
class SkillGroup:
   def __init__(self, realm):
      self.expCalc = experience.ExperienceCalculator()
      self.config  = realm.dataframe.config
      self.skills  = set()

   def update(self, realm, entity, actions):
      for skill in self.skills:
         skill.update(realm, entity)

   def packet(self):
      data = {}
      for skill in self.skills:
         data[skill.__class__.__name__.lower()] = skill.packet()

      return data

class Skill:
   skillItems = abc.ABCMeta

   def __init__(self, skillGroup):
      self.config  = skillGroup.config
      self.expCalc = skillGroup.expCalc
      self.exp     = 0

      skillGroup.skills.add(self)

   def packet(self):
      data = {}

      data['exp']   = self.exp
      data['level'] = self.level

      return data

   def update(self, realm, entity):
      pass

   def setExpByLevel(self, level):
      self.exp = self.expCalc.expAtLevel(level)

   @property
   def level(self):
      lvl = self.expCalc.levelAtExp(self.exp)
      assert lvl == int(lvl)
      return int(lvl)

### Skill Subsets ###
class Gathering(SkillGroup):
   def __init__(self, realm):
      super().__init__(realm)

      self.fishing      = Fishing(self)
      self.hunting      = Hunting(self)
      self.mining       = Mining(self)

class Processing(SkillGroup):
   def __init__(self, realm):
      super().__init__(realm)

      self.cooking      = Cooking(self)
      self.smithing     = Smithing(self)

class Combat(SkillGroup):
   def __init__(self, realm):
      super().__init__(realm)

      self.constitution = Constitution(self)
      self.defense      = Defense(self)
      self.melee        = Melee(self)
      self.range        = Range(self)
      self.mage         = Mage(self)

   def packet(self):
      data          = super().packet() 
      data['level'] = combat.level(self)

      return data

   def applyDamage(self, dmg, style):
      config = self.config
      scale = config.XP_SCALE
      self.constitution.exp += scale * dmg * config.CONSTITUTION_XP_SCALE

      skill = self.__dict__[style]
      skill.exp += scale * dmg * config.COMBAT_XP_SCALE

   def receiveDamage(self, dmg):
      scale = self.config.XP_SCALE
      self.constitution.exp += scale * dmg * 2
      self.defense.exp      += scale * dmg * 4


class Skills(Gathering, Processing, Combat):
   pass

### Individual Skills ###
class CombatSkill(Skill): pass

class Constitution(CombatSkill):
   def __init__(self, skillGroup):
      super().__init__(skillGroup)
      self.setExpByLevel(self.config.HEALTH)

   def update(self, realm, entity):
      health = entity.resources.health
      food   = entity.resources.food
      water  = entity.resources.water

      config = self.config

      # Heal if above fractional resource threshold
      foodThresh  = food > config.HEALTH_REGEN_THRESHOLD * entity.skills.hunting.level
      waterThresh = water > config.HEALTH_REGEN_THRESHOLD * entity.skills.fishing.level

      if foodThresh and waterThresh:
         restore = np.floor(self.level * self.config.HEALTH_RESTORE)
         health.increment(restore)

      if food.empty:
         health.decrement(1)

      if water.empty:
         health.decrement(1)

class Melee(CombatSkill): pass
class Range(CombatSkill): pass
class Mage(CombatSkill): pass
class Defense(CombatSkill): pass

class NonCombatSkill(Skill):
   def success(self, levelReq):
      level = self.level
      if level < levelReq:
         return False
      chance = 0.5 + 0.05*(level-levelReq)
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
   def __init__(self, skillGroup):
      super().__init__(skillGroup)
      self.setExpByLevel(self.config.RESOURCE)

   def update(self, realm, entity):
      water = entity.resources.water
      if entity.status.immune <= 0:
         water.decrement(1)

      if material.Water not in ai.utils.adjacentMats(
            realm.map.tiles, entity.pos):
         return

      restore = np.floor(self.level * self.config.RESOURCE_RESTORE)
      water.increment(restore)

      scale     = self.config.XP_SCALE
      self.exp += scale * restore


class Hunting(HarvestingSkill):
   def __init__(self, skillGroup):
      super().__init__(skillGroup)
      self.setExpByLevel(self.config.RESOURCE)

   def update(self, realm, entity):
      food = entity.resources.food
      if entity.status.immune <= 0:
         food.decrement(1)

      r, c = entity.pos
      if (type(realm.map.tiles[r, c].mat) not in [material.Forest] or
            not realm.map.harvest(r, c)):
         return

      restore = np.floor(self.level * self.config.RESOURCE_RESTORE)
      food.increment(restore)

      scale     = self.config.XP_SCALE
      self.exp += scale * restore


class Mining(HarvestingSkill): pass

class ProcessingSkill(NonCombatSkill):
   def process(self, inv, item):
      self.attempt(inv, item)

class Cooking(ProcessingSkill): pass
class Smithing(ProcessingSkill): pass
