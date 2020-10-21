from pdb import set_trace as T
import numpy as np

import random
from forge.blade.entity import entity, Player
from forge.blade.systems import combat, equipment, ai, combat
from forge.blade.lib.enums import Neon
from forge.blade.io.action import static as Action

#NPCs don't need exp or resources. Their skills are
#lighter weight but share a namespace. Eventually should
#refactor this to have player skills extend npc skills
class Skill:
   def __init__(self, level):
      self.level = level

   def packet(self):
      data = {}

      data['level'] = self.level
      data['exp']   = -1

      return data

class Constitution(Skill): pass
class Melee(Skill): pass
class Range(Skill): pass
class Mage(Skill): pass
class Defense(Skill): pass

#Add packet here
class Skills:
   def __init__(self, constitution, defense, melee, ranged, mage, style=None):
      self.constitution = Constitution(constitution)
      self.defense      = Defense(defense)
      self.melee        = Melee(melee)
      self.range        = Range(ranged)
      self.mage         = Mage(mage)
      self.style        = style

   def packet(self):
      data = {}

      data['constitution'] = self.constitution.packet()
      data['defense']      = self.defense.packet()
      data['melee']        = self.melee.packet()
      data['range']        = self.range.packet()
      data['mage']         = self.mage.packet()
      data['level']        = combat.level(self)

      return data

class Base(entity.Base):
   def __init__(self, ent, pos, iden, name, color):
      super().__init__(ent, pos, iden, name, color)
      self.name = name

   def update(self, realm, entity, actions):
      super().update(realm, entity, actions)
      if entity.health <= 0:
         self.killed = True
         return

   def packet(self):
      data = super().packet()

      return data

class NPC(entity.Entity):
   def __init__(self, config, iden, skills, loadout):
      super().__init__(config, iden)

      self.skills  = skills
      self.loadout = loadout

      self.status  = entity.Status(self)
      self.history = entity.History(self)

      self.health  = entity.Resource(self.skills.constitution.level)

   @property
   def alive(self):
      assert self.health >= 0
      if self.health == 0:
         return False
      return super().alive

   def step(self, realm, actions):
      super().step(realm, actions)

      if not self.alive:
         return 

      self.health.increment(1)
      self.lastAction = actions

      return actions

      for action, args in actions.items():
         args = args.values()
         action.call(realm, self, *args)

   def receiveDamage(self, source, dmg):
      self.history.damage = dmg

      if not self.alive:
         return

      self.health.decrement(dmg)

      if not self.alive and isinstance(source, Player):
         source.receiveLoot(self.loadout)

   @staticmethod
   def spawn(realm, pos, iden):
      config = realm.config
      wild   = combat.wilderness(config, pos)
      skills = Skills(*(NPC.levels(wild)))
      loadout = equipment.Loadout(
            chest = NPC.gearLevel(skills.defense.level),
            legs  = NPC.gearLevel(skills.defense.level))

      if wild < 33:
         return Passive(realm, pos, iden, skills, loadout)
      elif wild < 66:
         return PassiveAggressive(realm, pos, iden, skills, loadout)
      else:
         return Aggressive(realm, pos, iden, skills, loadout)
 
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

   #Generates random levels based on wilderness depth and clips within [1, 99]
   @staticmethod
   def levels(wild, alpha=0.8):
      mmin = int(alpha * wild)
      mmax = int((1+alpha) * wild)
 
      if mmax < mmin:
         mmax = mmin
         
      constitution = NPC.clippedLevel(mmin, mmax)
      defense      = NPC.clippedLevel(mmin, mmax)
      attack       = NPC.clippedLevel(mmin, mmax)

      melee = ranged = mage = 1
      idx   = random.randint(0, 4)
      if idx <=2:
         melee = attack
         style = Action.Melee
      elif idx == 3:
         ranged = attack
         style  = Action.Range
      else:
         mage   = attack
         style  = Action.Mage

      return constitution, defense, melee, ranged, mage, style

   def packet(self):
      data = super().packet()

      data['base']     = self.base.packet()      
      data['skills']   = self.skills.packet()      
      data['resource'] = {'health': self.health.packet()}

      return data

class Passive(NPC):
   def __init__(self, config, pos, iden, skills, loadout):
      super().__init__(config, iden, skills, loadout)
      self.base = Base(self, pos, iden, "Passive", Neon.GREEN)

   #NPCs have direct access to the world
   def step(self, realm):
      actions = ai.policy.passive(realm, self)
      return super().step(realm, actions)

class PassiveAggressive(NPC):
   def __init__(self, config, pos, iden, skills, loadout):
      super().__init__(config, iden, skills, loadout)
      self.base   = Base(self, pos, iden, "Neutral", Neon.ORANGE)

   def step(self, realm):
      actions = ai.policy.neutral(realm, self)
      return super().step(realm, actions)

class Aggressive(NPC):
   def __init__(self, config, pos, iden, skills, loadout):
      super().__init__(config, iden, skills, loadout)
      self.base   = Base(self, pos, iden, "Hostile", Neon.RED)
      self.vision = int(max(self.vision, 1 + combat.level(self.skills) // 10))

   def step(self, realm):
      actions = ai.policy.hostile(realm, self)
      return super().step(realm, actions)
