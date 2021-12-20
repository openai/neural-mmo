from pdb import set_trace as T
import numpy as np

import random

import nmmo
from nmmo.entity import entity
from nmmo.systems import combat, equipment, ai, combat, skill
from nmmo.lib.colors import Neon

class NPC(entity.Entity):
   def __init__(self, realm, pos, iden, name, color, pop):
      super().__init__(realm, pos, iden, name, color, pop)
      self.skills = skill.Combat(self)

   def update(self, realm, actions):
      super().update(realm, actions)

      if not self.alive:
         return

      self.resources.health.increment(1)
      self.lastAction = actions

   @staticmethod
   def spawn(realm, pos, iden):
      config = realm.config

      #Select AI Policy
      danger = combat.danger(config, pos)
      if danger >= config.NPC_SPAWN_AGGRESSIVE:
         ent = Aggressive(realm, pos, iden)
      elif danger >= config.NPC_SPAWN_NEUTRAL:
         ent = PassiveAggressive(realm, pos, iden)
      elif danger >= config.NPC_SPAWN_PASSIVE:
         ent = Passive(realm, pos, iden)
      else:
         return

      #Set levels
      levels = NPC.clippedLevels(config, danger, n=5)
      constitution, defense, melee, ranged, mage = levels

      ent.resources.health.max = constitution
      ent.resources.health.update(constitution)

      ent.skills.constitution.setExpByLevel(constitution)
      ent.skills.defense.setExpByLevel(defense)
      ent.skills.melee.setExpByLevel(melee)
      ent.skills.range.setExpByLevel(ranged)
      ent.skills.mage.setExpByLevel(mage)

      ent.skills.style = random.choice(
         (nmmo.action.Melee, nmmo.action.Range, nmmo.action.Mage))

      #Set equipment levels
      ent.loadout.chestplate.level = NPC.gearLevel(defense)
      ent.loadout.platelegs.level  = NPC.gearLevel(defense)

      return ent

   def yieldDrops(self):
      self.lastAttacker.receiveDrops(self.drops.roll())

   @staticmethod
   def gearLevel(lvl, offset=10):
      proposed = random.gauss(lvl-offset, offset)
      lvl      = np.clip(proposed, 0, lvl)
      return int(lvl)

   @staticmethod
   def clippedLevels(config, danger, n=1):
      lmin    = config.NPC_LEVEL_MIN
      lmax    = config.NPC_LEVEL_MAX

      lbase   = danger*(lmax-lmin) + lmin
      lspread = config.NPC_LEVEL_SPREAD

      lvlMin  = int(max(lmin, lbase - lspread))
      lvlMax  = int(min(lmax, lbase + lspread))

      lvls = [random.randint(lvlMin, lvlMax) for _ in range(n)]

      if n == 1:
         return lvls[0]

      return lvls
 
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
      self.vision = int(max(self.vision, 1 + combat.level(self.skills) // 10))
      self.dataframe.init(nmmo.Serialized.Entity, self.entID, self.pos)

   def decide(self, realm):
      return ai.policy.hostile(realm, self)
