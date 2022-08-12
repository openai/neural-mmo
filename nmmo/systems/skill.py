from pdb import set_trace as T
import logging
import abc

import numpy as np
from nmmo.io.stimulus import Serialized
from nmmo.systems import experience, combat, ai
from nmmo.lib import material

### Infrastructure ###
class SkillGroup:
   def __init__(self, realm, entity):
      self.config  = realm.config
      self.realm   = realm

      self.expCalc = experience.ExperienceCalculator()
      self.skills  = set()

   def update(self, realm, entity):
       for skill in self.skills:
           skill.update(realm, entity)

   def packet(self):
      data = {}
      for skill in self.skills:
         data[skill.__class__.__name__.lower()] = skill.packet()

      return data

class Skill:
   skillItems = abc.ABCMeta

   def __init__(self, realm, entity, skillGroup):
      self.config  = realm.config
      self.realm   = realm

      self.expCalc = skillGroup.expCalc
      self.exp     = 0

      skillGroup.skills.add(self)

   def packet(self):
      data = {}

      data['exp']   = self.exp
      data['level'] = self.level.val

      return data

   def add_xp(self, xp):
      level     = self.expCalc.levelAtExp(self.exp)
      self.exp += xp * self.config.PROGRESSION_BASE_XP_SCALE

      level = self.expCalc.levelAtExp(self.exp)
      self.level.update(int(level))

      if self.config.LOG_MILESTONES and self.realm.quill.milestone.log_max(f'Level_{self.__class__.__name__}', level) and self.config.LOG_VERBOSE:
         logging.info(f'PROGRESSION: Reached level {level} {self.__class__.__name__}')

   def setExpByLevel(self, level):
      self.exp = self.expCalc.expAtLevel(level)
      self.level.update(int(level))

### Skill Bases ###
class CombatSkill(Skill):
  def update(self, realm, entity):
     pass

class NonCombatSkill(Skill): pass


class HarvestSkill(NonCombatSkill):
    def processDrops(self, realm, entity, matl, dropTable):
        level = 1
        tool  = entity.equipment.held
        if type(tool) == matl.tool:
            level = tool.level.val

        #TODO: double-check drop table quantity
        for drop in dropTable.roll(realm, level):
            assert drop.level.val == level, 'Drop level does not match roll specification'

            if self.config.LOG_MILESTONES and realm.quill.milestone.log_max(f'Gather_{drop.__class__.__name__}', level) and self.config.LOG_VERBOSE:
                logging.info(f'PROFESSION: Gathered level {level} {drop.__class__.__name__} (level {self.level.val} {self.__class__.__name__})') 

            if entity.inventory.space:
                entity.inventory.receive(drop)

    def harvest(self, realm, entity, matl, deplete=True):
        r, c = entity.pos
        if realm.map.tiles[r, c].state != matl:
            return

        dropTable = realm.map.harvest(r, c, deplete)
        if dropTable:
            self.processDrops(realm, entity, matl, dropTable)
            return True

    def harvestAdjacent(self, realm, entity, matl, deplete=True):
        r, c      = entity.pos
        dropTable = None

        if realm.map.tiles[r-1, c].state == matl:
            dropTable = realm.map.harvest(r-1, c, deplete)
        if realm.map.tiles[r+1, c].state == matl:
            dropTable = realm.map.harvest(r+1, c, deplete)
        if realm.map.tiles[r, c-1].state == matl:
            dropTable = realm.map.harvest(r, c-1, deplete)
        if realm.map.tiles[r, c+1].state == matl:
            dropTable = realm.map.harvest(r, c+1, deplete)

        if dropTable:
            self.processDrops(realm, entity, matl, dropTable)
            return True

class AmmunitionSkill(HarvestSkill):
    def processDrops(self, realm, entity, matl, dropTable):
        super().processDrops(realm, entity, matl, dropTable)

        self.add_xp(self.config.PROGRESSION_AMMUNITION_XP_SCALE)


class ConsumableSkill(HarvestSkill):
    def processDrops(self, realm, entity, matl, dropTable):
        super().processDrops(realm, entity, matl, dropTable)

        self.add_xp(self.config.PROGRESSION_CONSUMABLE_XP_SCALE)


### Skill groups ###
class Basic(SkillGroup):
    def __init__(self, realm, entity):
        super().__init__(realm, entity)

        self.water = Water(realm, entity, self)
        self.food  = Food(realm, entity, self)

    @property
    def basicLevel(self):
        return 0.5 * (self.water.level
                + self.food.level)

class Harvest(SkillGroup):
    def __init__(self, realm, entity):
        super().__init__(realm, entity)

        self.fishing      = Fishing(realm, entity, self)
        self.herbalism    = Herbalism(realm, entity, self)
        self.prospecting  = Prospecting(realm, entity, self)
        self.carving      = Carving(realm, entity, self)
        self.alchemy      = Alchemy(realm, entity, self)

    @property
    def harvestLevel(self):
        return max(self.fishing.level,
                   self.herbalism.level,
                   self.prospecting.level,
                   self.carving.level,
                   self.alchemy.level) 

class Combat(SkillGroup):
   def __init__(self, realm, entity):
      super().__init__(realm, entity)

      self.melee = Melee(realm, entity, self)
      self.range = Range(realm, entity, self)
      self.mage  = Mage(realm, entity, self)

   def packet(self):
      data          = super().packet() 
      data['level'] = combat.level(self)

      return data

   @property
   def combatLevel(self):
      return max(self.melee.level,
                 self.range.level,
                 self.mage.level)

   def applyDamage(self, dmg, style):
      if not self.config.PROGRESSION_SYSTEM_ENABLED:
         return

      config = self.config
      skill  = self.__dict__[style]
      skill.add_xp(config.PROGRESSION_COMBAT_XP_SCALE)

   def receiveDamage(self, dmg):
      pass

class Skills(Basic, Harvest, Combat):
    pass

### Skills ###
class Melee(CombatSkill):
    def __init__(self, realm, ent, skillGroup):
        self.level = Serialized.Entity.Melee(ent.dataframe, ent.entID)
        super().__init__(realm, ent, skillGroup)

class Range(CombatSkill):
    def __init__(self, realm, ent, skillGroup):
        self.level = Serialized.Entity.Range(ent.dataframe, ent.entID)
        super().__init__(realm, ent, skillGroup)

class Mage(CombatSkill):
    def __init__(self, realm, ent, skillGroup):
        self.level = Serialized.Entity.Mage(ent.dataframe, ent.entID)
        super().__init__(realm, ent, skillGroup)

Melee.weakness = Mage
Range.weakness = Melee
Mage.weakness  = Range

### Individual Skills ###

class CombatSkill(Skill): pass

class Lvl:
    def __init__(self, val):
        self.val = val

    def update(self, val):
        self.val = val

class Water(HarvestSkill):
    def __init__(self, realm, entity, skillGroup):
        self.level = Lvl(1)
        super().__init__(realm, entity, skillGroup)

    def update(self, realm, entity):
        config = self.config
        if not config.RESOURCE_SYSTEM_ENABLED:
            return

        depletion = config.RESOURCE_DEPLETION_RATE
        water = entity.resources.water
        water.decrement(depletion)

        tiles = realm.map.tiles
        if not self.harvestAdjacent(realm, entity, material.Water, deplete=False):
            return

        restore = np.floor(config.RESOURCE_BASE
                         * config.RESOURCE_HARVEST_RESTORE_FRACTION)
        water.increment(restore)

class Food(HarvestSkill):
    def __init__(self, realm, entity, skillGroup):
        self.level = Lvl(1)
        super().__init__(realm, entity, skillGroup)

    def update(self, realm, entity):
        config = self.config
        if not config.RESOURCE_SYSTEM_ENABLED:
            return

        depletion = config.RESOURCE_DEPLETION_RATE
        food = entity.resources.food
        food.decrement(depletion)

        if not self.harvest(realm, entity, material.Forest):
            return

        restore = np.floor(config.RESOURCE_BASE
                         * config.RESOURCE_HARVEST_RESTORE_FRACTION)
        food.increment(restore)

class Fishing(ConsumableSkill):
    def __init__(self, realm, ent, skillGroup):
        self.level = Serialized.Entity.Fishing(ent.dataframe, ent.entID)
        super().__init__(realm, ent, skillGroup)

    def update(self, realm, entity):
        self.harvestAdjacent(realm, entity, material.Fish)

class Herbalism(ConsumableSkill):
    def __init__(self, realm, ent, skillGroup):
        self.level = Serialized.Entity.Herbalism(ent.dataframe, ent.entID)
        super().__init__(realm, ent, skillGroup)

    def update(self, realm, entity):
        self.harvest(realm, entity, material.Herb)

class Prospecting(AmmunitionSkill):
    def __init__(self, realm, ent, skillGroup):
        self.level = Serialized.Entity.Prospecting(ent.dataframe, ent.entID)
        super().__init__(realm, ent, skillGroup)

    def update(self, realm, entity):
        self.harvest(realm, entity, material.Ore)

class Carving(AmmunitionSkill):
    def __init__(self, realm, ent, skillGroup):
        self.level = Serialized.Entity.Carving(ent.dataframe, ent.entID)
        super().__init__(realm, ent, skillGroup)

    def update(self, realm, entity):
        self.harvest(realm, entity, material.Tree)

class Alchemy(AmmunitionSkill):
    def __init__(self, realm, ent, skillGroup):
        self.level = Serialized.Entity.Alchemy(ent.dataframe, ent.entID)
        super().__init__(realm, ent, skillGroup)

    def update(self, realm, entity):
        self.harvest(realm, entity, material.Crystal)
