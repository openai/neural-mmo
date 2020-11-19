from pdb import set_trace as T
import numpy as np

import random
from forge.blade.entity import entity, Player
from forge.blade.systems import combat, equipment, ai, combat, skill
from forge.blade.lib.enums import Neon
from forge.blade.io.action import static as Action
from forge.blade.io.stimulus import Static

class NPC(entity.Entity):
   def __init__(self, realm, pos, iden, name, color, pop):
      super().__init__(realm, pos, iden, name, color, pop)
      self.skills = skill.Combat(self)

   def update(self, realm, actions):
      if not super().update(realm, actions):
         return

      self.resources.health.increment(1)
      self.lastAction = actions

   @staticmethod
   def spawn(realm, pos, iden, alpha=0.8):
      config = realm.config
      wild   = combat.wilderness(config, pos) + 1

      #Select AI Policy
      if wild < 33:
         ent = Passive(realm, pos, iden)
      elif wild < 66:
         ent = PassiveAggressive(realm, pos, iden)
      else:
         ent = Aggressive(realm, pos, iden)

      #Generates combat levels based on wilderness depth and clips within [1, 99]
      mmin = int(alpha * wild)
      mmax = int((1+alpha) * wild)
        
      ent.skills.constitution.setExpByLevel(NPC.clippedLevel(mmin, mmax))
      ent.skills.defense.setExpByLevel(NPC.clippedLevel(mmin, mmax))

      attack = NPC.clippedLevel(mmin, mmax)
      idx    = random.randint(0, 4)

      if idx <=2:
         ent.skills.melee.setExpByLevel(attack)
         ent.skills.style = Action.Melee
      elif idx == 3:
         ent.skills.range.setExpByLevel(attack)
         ent.skills.style = Action.Range
      else:
         ent.skills.mage.setExpByLevel(attack)
         ent.skills.style = Action.Mage

      #Set equipment levels
      defense = ent.skills.defense.level
      ent.loadout.chestplate.level = NPC.gearLevel(defense)
      ent.loadout.platelegs.level  = NPC.gearLevel(defense)

      return ent

   def yieldDrops(self):
      self.lastAttacker.receiveDrops(self.drops.roll())

   @staticmethod
   def gearLevel(lvl, offset=10):
      proposed = random.gauss(lvl-offset, offset)
      lvl      = min(proposed, lvl)
      lvl      = max(0, lvl)
      return int(lvl)

   @staticmethod
   def clippedLevel(mmin, mmax):
      lvl = random.randint(mmin, mmax)
      lvl = min(99, lvl)
      lvl = max(1, lvl)
      return lvl

   def packet(self):
      data = super().packet()

      data['base']     = self.base.packet()      
      data['skills']   = self.skills.packet()      
      data['resource'] = {'health': self.resources.health.packet()}

      return data

class Passive(NPC):
   def __init__(self, realm, pos, iden):
      super().__init__(realm, pos, iden, 'Passive', Neon.GREEN, -1)
      self.dataframe.init(Static.Entity, iden, pos)

   #NPCs have direct access to the world
   def decide(self, realm):
      return ai.policy.passive(realm, self)

class PassiveAggressive(NPC):
   def __init__(self, realm, pos, iden):
      super().__init__(realm, pos, iden, 'Neutral', Neon.ORANGE, -2)
      self.dataframe.init(Static.Entity, iden, pos)

   def decide(self, realm):
      return ai.policy.neutral(realm, self)

class Aggressive(NPC):
   def __init__(self, realm, pos, iden):
      super().__init__(realm, pos, iden, 'Hostile', Neon.RED, -3)
      self.dataframe.init(Static.Entity, iden, pos)
      self.vision = int(max(self.vision, 1 + combat.level(self.skills) // 10))
      self.dataframe.init(Static.Entity, self.entID, self.pos)

   def decide(self, realm):
      return ai.policy.hostile(realm, self)
