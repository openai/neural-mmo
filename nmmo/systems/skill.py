from pdb import set_trace as T
import abc

import numpy as np
from nmmo.io.stimulus import Serialized
from nmmo.systems import experience, combat, ai
from nmmo.lib import material

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

   def __init__(self, entity, skillGroup):
      self.config  = skillGroup.config
      self.expCalc = skillGroup.expCalc
      self.exp     = 0

      skillGroup.skills.add(self)

   def packet(self):
      data = {}

      data['exp']   = self.exp
      data['level'] = self.level.val

      return data

   def update(self, realm, entity):
      scale     = self.config.XP_SCALE
      self.exp += scale * xp

   def setExpByLevel(self, level):
      self.exp = self.expCalc.expAtLevel(level)
      self.level.update(int(level))

### Skill Bases ###
class CombatSkill(Skill):
    def update(self, realm, entity):
        pass

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
            # Check that everything is available
            if not inv.satisfies(item.recipe): return
            inv.removeRecipe(item.recipe)

        if item.alwaysSucceeds or self.success(item.createLevel):
            inv.add(item, item.amtMade)
            self.exp += item.exp
            return True

class HarvestSkill(NonCombatSkill):
    def processDrops(self, realm, entity, matl, dropTable):
        level = 0
        tool  = entity.equipment.held
        if type(tool) == matl.tool:
            level = tool.level.val

        for drop in dropTable.roll(realm, self.level.val):
            if entity.inventory.space:
                entity.inventory.receive(drop)

        self.exp += (self.config.PROGRESSION_BASE_XP_SCALE
                  * self.config.PROGRESSION_HARVEST_XP_SCALE)

        level = self.expCalc.levelAtExp(self.exp)
        self.level.update(int(level))

    def harvest(self, realm, entity, matl, deplete=True):
        r, c = entity.pos
        if realm.map.tiles[r, c].state != matl:
            return

        if dropTable := realm.map.harvest(r, c, deplete):
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

### Skill groups ###
class Basic(SkillGroup):
    def __init__(self, entity):
        super().__init__(entity)

        self.water = Water(entity, self)
        self.food  = Food(entity, self)

    @property
    def basicLevel(self):
        return 0.5 * (self.water.level
                + self.food.level)

class Harvest(SkillGroup):
    def __init__(self, entity):
        super().__init__(entity)

        self.fishing      = Fishing(entity, self)
        self.herbalism    = Herbalism(entity, self)
        self.prospecting  = Prospecting(entity, self)
        self.carving      = Carving(entity, self)
        self.alchemy      = Alchemy(entity, self)

    @property
    def harvestLevel(self):
        return max(self.fishing.level,
                   self.herbalism.level,
                   self.prospecting.level,
                   self.carving.level,
                   self.alchemy.level) 

class Combat(SkillGroup):
   def __init__(self, entity):
      super().__init__(entity)

      self.melee = Melee(entity, self)
      self.range = Range(entity, self)
      self.mage  = Mage(entity, self)

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
      if not self.config.game_system_enabled('Progression'):
         return

      config    = self.config
      scale     = (config.PROGRESSION_BASE_XP_SCALE
                 * config.PROGRESSION_COMBAT_XP_SCALE)

      skill      = self.__dict__[style]
      skill.exp += scale * dmg

      level = self.expCalc.levelAtExp(skill.exp)
      skill.level.update(int(level))

   def receiveDamage(self, dmg):
      pass

class Skills(Basic, Harvest, Combat):
    pass

### Skills ###
class Melee(CombatSkill):
    def __init__(self, ent, skillGroup):
        self.level = Serialized.Entity.Melee(ent.dataframe, ent.entID)
        super().__init__(ent, skillGroup)

class Range(CombatSkill):
    def __init__(self, ent, skillGroup):
        self.level = Serialized.Entity.Range(ent.dataframe, ent.entID)
        super().__init__(ent, skillGroup)

class Mage(CombatSkill):
    def __init__(self, ent, skillGroup):
        self.level = Serialized.Entity.Mage(ent.dataframe, ent.entID)
        super().__init__(ent, skillGroup)

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
    def __init__(self, entity, skillGroup):
        self.level = Lvl(1)
        super().__init__(entity, skillGroup)

    def update(self, realm, entity):
        config = self.config
        if not config.game_system_enabled('Resource'):
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
    def __init__(self, entity, skillGroup):
        self.level = Lvl(1)
        super().__init__(entity, skillGroup)

    def update(self, realm, entity):
        config = self.config
        if not config.game_system_enabled('Resource'):
            return

        depletion = config.RESOURCE_DEPLETION_RATE
        food = entity.resources.food
        food.decrement(depletion)

        if not self.harvest(realm, entity, material.Forest):
            return

        restore = np.floor(config.RESOURCE_BASE
                         * config.RESOURCE_HARVEST_RESTORE_FRACTION)
        food.increment(restore)

class Fishing(HarvestSkill):
    def __init__(self, ent, skillGroup):
        self.level = Serialized.Entity.Fishing(ent.dataframe, ent.entID)
        super().__init__(ent, skillGroup)

    def update(self, realm, entity):
        self.harvestAdjacent(realm, entity, material.Fish)

class Herbalism(HarvestSkill):
    def __init__(self, ent, skillGroup):
        self.level = Serialized.Entity.Herbalism(ent.dataframe, ent.entID)
        super().__init__(ent, skillGroup)

    def update(self, realm, entity):
        self.harvest(realm, entity, material.Herb)

class Prospecting(HarvestSkill):
    def __init__(self, ent, skillGroup):
        self.level = Serialized.Entity.Prospecting(ent.dataframe, ent.entID)
        super().__init__(ent, skillGroup)

    def update(self, realm, entity):
        self.harvest(realm, entity, material.Ore)

class Carving(HarvestSkill):
    def __init__(self, ent, skillGroup):
        self.level = Serialized.Entity.Carving(ent.dataframe, ent.entID)
        super().__init__(ent, skillGroup)

    def update(self, realm, entity):
        self.harvest(realm, entity, material.Tree)

class Alchemy(HarvestSkill):
    def __init__(self, ent, skillGroup):
        self.level = Serialized.Entity.Alchemy(ent.dataframe, ent.entID)
        super().__init__(ent, skillGroup)

    def update(self, realm, entity):
        self.harvest(realm, entity, material.Crystal)
