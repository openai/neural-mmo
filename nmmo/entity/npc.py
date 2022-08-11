from pdb import set_trace as T
import numpy as np

import random

import nmmo
from nmmo.entity import entity
from nmmo.systems import combat, equipment, ai, combat, skill
from nmmo.lib.colors import Neon
from nmmo.systems import item as Item
from nmmo.systems import droptable
from nmmo.io import action as Action


class Equipment:
   def __init__(self, total,
           melee_attack, range_attack, mage_attack,
           melee_defense, range_defense, mage_defense):

      self.level         = total
      self.ammunition    = None

      self.melee_attack  = melee_attack
      self.range_attack  = range_attack
      self.mage_attack   = mage_attack
      self.melee_defense = melee_defense
      self.range_defense = range_defense
      self.mage_defense  = mage_defense

   def total(self, getter):
      return getter(self)

   @property
   def packet(self):
      packet = {}

      packet['item_level']    = self.total

      packet['melee_attack']  = self.melee_attack
      packet['range_attack']  = self.range_attack
      packet['mage_attack']   = self.mage_attack
      packet['melee_defense'] = self.melee_defense
      packet['range_defense'] = self.range_defense
      packet['mage_defense']  = self.mage_defense

      return packet


class NPC(entity.Entity):
   def __init__(self, realm, pos, iden, name, color, pop):
      super().__init__(realm, pos, iden, name, color, pop)
      self.skills = skill.Combat(realm, self)
      self.realm = realm
      

   def update(self, realm, actions):
      super().update(realm, actions)

      if not self.alive:
         return

      self.resources.health.increment(1)
      self.lastAction = actions

   def receiveDamage(self, source, dmg):
       if super().receiveDamage(source, dmg):
           return True

       for item in self.droptable.roll(self.realm, self.level):
           if source.inventory.space:
               source.inventory.receive(item)

   @staticmethod
   def spawn(realm, pos, iden):
      config = realm.config

      # Select AI Policy
      danger = combat.danger(config, pos)
      if danger >= config.NPC_SPAWN_AGGRESSIVE:
         ent = Aggressive(realm, pos, iden)
      elif danger >= config.NPC_SPAWN_NEUTRAL:
         ent = PassiveAggressive(realm, pos, iden)
      elif danger >= config.NPC_SPAWN_PASSIVE:
         ent = Passive(realm, pos, iden)
      else:
         return

      ent.spawn_danger = danger

      # Select combat focus
      style = random.choice((Action.Melee, Action.Range, Action.Mage))
      ent.skills.style = style

      # Compute level
      level = 0
      if config.PROGRESSION_SYSTEM_ENABLED:
          level_min = config.NPC_LEVEL_MIN
          level_max = config.NPC_LEVEL_MAX
          level     = int(danger * (level_max - level_min) + level_min)

          # Set skill levels
          if style == Action.Melee:
              ent.skills.melee.setExpByLevel(level)
          elif style == Action.Range:
              ent.skills.range.setExpByLevel(level)
          elif style == Action.Mage:
              ent.skills.mage.setExpByLevel(level)

      # Gold
      if config.EXCHANGE_SYSTEM_ENABLED:
          ent.inventory.gold.quantity.update(level)

      ent.droptable = droptable.Standard()

      # Equipment to instantiate
      if config.EQUIPMENT_SYSTEM_ENABLED:
          lvl     = level - random.random()
          ilvl    = int(5 * lvl)

          offense = int(config.NPC_BASE_DAMAGE + lvl*config.NPC_LEVEL_DAMAGE)
          defense = int(config.NPC_BASE_DEFENSE + lvl*config.NPC_LEVEL_DEFENSE)

          ent.equipment = Equipment(ilvl, offense, offense, offense, defense, defense, defense)

          armor =  [Item.Hat, Item.Top, Item.Bottom]
          ent.droptable.add(random.choice(armor))

      if config.PROFESSION_SYSTEM_ENABLED:
         tools =  [Item.Rod, Item.Gloves, Item.Pickaxe, Item.Chisel, Item.Arcane]
         ent.droptable.add(random.choice(tools))

      return ent 

   def packet(self):
      data = super().packet()

      data['base']     = self.base.packet()      
      data['skills']   = self.skills.packet()      
      data['resource'] = {'health': self.resources.health.packet()}

      return data

   @property
   def isNPC(self) -> bool:
      return True

class Passive(NPC):
   def __init__(self, realm, pos, iden):
      super().__init__(realm, pos, iden, 'Passive', Neon.GREEN, -1)
      self.dataframe.init(nmmo.Serialized.Entity, iden, pos)

   def decide(self, realm):
      return ai.policy.passive(realm, self)

class PassiveAggressive(NPC):
   def __init__(self, realm, pos, iden):
      super().__init__(realm, pos, iden, 'Neutral', Neon.ORANGE, -2)
      self.dataframe.init(nmmo.Serialized.Entity, iden, pos)

   def decide(self, realm):
      return ai.policy.neutral(realm, self)

class Aggressive(NPC):
   def __init__(self, realm, pos, iden):
      super().__init__(realm, pos, iden, 'Hostile', Neon.RED, -3)
      self.dataframe.init(nmmo.Serialized.Entity, iden, pos)

   def decide(self, realm):
      return ai.policy.hostile(realm, self)
