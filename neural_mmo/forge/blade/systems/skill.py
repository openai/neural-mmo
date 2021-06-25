from pdb import set_trace as T
import abc

import numpy as np
from neural_mmo.forge.blade.systems import experience, combat, ai

from neural_mmo.forge.blade.lib import material

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
class Harvesting(SkillGroup):
   def __init__(self, realm):
      super().__init__(realm)

      self.fishing      = Fishing(self)
      self.hunting      = Hunting(self)

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
      if not self.config.game_system_enabled('Progression'):
         return

      config    = self.config
      baseScale = config.PROGRESSION_BASE_XP_SCALE
      combScale = config.PROGRESSION_COMBAT_XP_SCALE
      conScale  = config.PROGRESSION_CONSTITUTION_XP_SCALE

      self.constitution.exp += dmg * baseScale * conScale

      skill      = self.__dict__[style]
      skill.exp += dmg * baseScale * combScale

   def receiveDamage(self, dmg):
      if not self.config.game_system_enabled('Progression'):
         return

      config    = self.config
      baseScale = config.PROGRESSION_BASE_XP_SCALE
      combScale = config.PROGRESSION_COMBAT_XP_SCALE
      conScale  = config.PROGRESSION_CONSTITUTION_XP_SCALE

      self.constitution.exp += dmg * baseScale * conScale
      self.defense.exp      += dmg * baseScale * combScale

class Skills(Harvesting, Combat):
   pass

### Individual Skills ###
class CombatSkill(Skill): pass

class Constitution(CombatSkill):
   def __init__(self, skillGroup):
      super().__init__(skillGroup)
      self.setExpByLevel(self.config.BASE_HEALTH)

   def update(self, realm, entity):
      health = entity.resources.health
      food   = entity.resources.food
      water  = entity.resources.water
      config = self.config

      if not config.game_system_enabled('Resource'):
         health.increment(1)
         return

      # Heal if above fractional resource threshold
      regen       = config.RESOURCE_HEALTH_REGEN_THRESHOLD
      foodThresh  = food > regen * entity.skills.hunting.level
      waterThresh = water > regen * entity.skills.fishing.level

      if foodThresh and waterThresh:
         restore = config.RESOURCE_HEALTH_RESTORE_FRACTION
         restore = np.floor(restore * self.level)
         health.increment(restore)

      if food.empty:
         health.decrement(1)

      if water.empty:
         health.decrement(1)

class Melee(CombatSkill): pass
class Range(CombatSkill): pass
class Mage(CombatSkill): pass
class Defense(CombatSkill): pass

class Fishing(Skill):
   def __init__(self, skillGroup):
      super().__init__(skillGroup)
      config, level = self.config, 1
      if config.game_system_enabled('Progression'):
         level = config.PROGRESSION_BASE_RESOURCE
      elif config.game_system_enabled('Resource'):
         level = config.RESOURCE_BASE_RESOURCE

      self.setExpByLevel(level)

   def update(self, realm, entity):
      if not self.config.game_system_enabled('Resource'):
         return

      water = entity.resources.water
      water.decrement(1)

      if material.Water not in ai.utils.adjacentMats(
            realm.map.tiles, entity.pos):
         return

      restore = self.config.RESOURCE_HARVEST_RESTORE_FRACTION
      restore = np.floor(restore * self.level)
      water.increment(restore)

      if self.config.game_system_enabled('Progression'):
         self.exp += self.config.PROGRESSION_BASE_XP_SCALE * restore

class Hunting(Skill):
   def __init__(self, skillGroup):
      super().__init__(skillGroup)
      config, level = self.config, 1
      if config.game_system_enabled('Progression'):
         level = config.PROGRESSION_BASE_RESOURCE
      elif config.game_system_enabled('Resource'):
         level = config.RESOURCE_BASE_RESOURCE

      self.setExpByLevel(level)

   def update(self, realm, entity):
      if not self.config.game_system_enabled('Resource'):
         return

      food = entity.resources.food
      food.decrement(1)

      r, c = entity.pos
      if (type(realm.map.tiles[r, c].mat) not in [material.Forest] or
            not realm.map.harvest(r, c)):
         return

      restore = self.config.RESOURCE_HARVEST_RESTORE_FRACTION
      restore = np.floor(restore * self.level)
      food.increment(restore)

      if self.config.game_system_enabled('Progression'):
         self.exp += self.config.PROGRESSION_BASE_XP_SCALE * restore
