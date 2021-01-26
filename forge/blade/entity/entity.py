from pdb import set_trace as T
import numpy as np

from forge.blade.systems import skill, droptable, combat, equipment
from forge.blade.lib import material

from forge.blade.io.action import static as Action
from forge.blade.io.stimulus import Static

class Resources:
   def __init__(self, ent):
      self.health = Static.Entity.Health(ent.dataframe, ent.entID)
      self.water  = Static.Entity.Water( ent.dataframe, ent.entID)
      self.food   = Static.Entity.Food(  ent.dataframe, ent.entID)

   def update(self, realm, entity, actions):
      self.health.max = entity.skills.constitution.level
      self.water.max  = entity.skills.fishing.level
      self.food.max   = entity.skills.hunting.level

   def packet(self):
      data = {}
      data['health'] = self.health.packet()
      data['food']   = self.food.packet()
      data['water']  = self.water.packet()
      return data

class Status:
   def __init__(self, ent):
      self.config = ent.config

      self.wilderness = Static.Entity.Wilderness(ent.dataframe, ent.entID)
      self.immune     = Static.Entity.Immune(    ent.dataframe, ent.entID)
      self.freeze     = Static.Entity.Freeze(    ent.dataframe, ent.entID)

   def update(self, realm, entity, actions):
      self.immune.decrement()
      self.freeze.decrement()

      wilderness = combat.wilderness(self.config, entity.pos)
      self.wilderness.update(wilderness)

   def packet(self):
      data = {}
      data['wilderness'] = self.wilderness.val
      data['immune']     = self.immune.val
      data['freeze']     = self.freeze.val
      return data

class History:
   def __init__(self, ent):
      self.actions = None
      self.attack  = None

      self.damage    = Static.Entity.Damage(   ent.dataframe, ent.entID)
      self.timeAlive = Static.Entity.TimeAlive(ent.dataframe, ent.entID)

      self.lastPos = None

   def update(self, realm, entity, actions):
      self.attack  = None
      self.actions = actions
      self.damage.update(0)
      self.timeAlive.increment()

   def packet(self):
      data = {}
      data['damage']    = self.damage.val
      data['timeAlive'] = self.timeAlive.val

      if self.attack is not None:
         data['attack'] = self.attack

      return data

class Base:
   def __init__(self, ent, pos, iden, name, color, pop):
      self.name  = name + str(iden)
      self.color = color
      r, c       = pos

      self.r          = Static.Entity.R(ent.dataframe, ent.entID, r)
      self.c          = Static.Entity.C(ent.dataframe, ent.entID, c)

      self.population = Static.Entity.Population(ent.dataframe, ent.entID, pop)
      self.self       = Static.Entity.Self(      ent.dataframe, ent.entID, True)

      ent.dataframe.init(Static.Entity, ent.entID, (r, c))

   def update(self, realm, entity, actions):
      r, c = self.pos
      if realm.map.tiles[r, c].lava:
         entity.receiveDamage(None, entity.resources.health.val)

      if entity.resources.health.empty:
         self.killed = True
         return False

      return True

   @property
   def pos(self):
      return self.r.val, self.c.val

   def packet(self):
      data = {}

      data['r']          = self.r.val
      data['c']          = self.c.val
      data['name']       = self.name
      data['color']      = self.color.packet()
      data['population'] = self.population.val
      data['self']       = self.self.val

      return data

class Entity:
   def __init__(self, realm, pos, iden, name, color, pop):
      self.dataframe    = realm.dataframe
      self.config       = realm.config
      self.entID        = iden

      self.repr         = None
      self.killed       = False
      self.vision       = 5

      self.attacker     = None
      self.target       = None
      self.closest      = None
      self.spawnPos     = pos

      #Submodules
      self.base      = Base(self, pos, iden, name, color, pop)
      self.status    = Status(self)
      self.history   = History(self)
      self.resources = Resources(self)
      self.loadout   = equipment.Loadout()

   def packet(self):
      data = {}

      data['status']  = self.status.packet()
      data['history'] = self.history.packet()
      data['loadout'] = self.loadout.packet()
      data['alive']   = self.alive

      return data

   def update(self, realm, actions):
      '''Update occurs after actions, e.g. does not include history'''
      if self.history.damage == 0:
         self.attacker = None

      self.base.update(realm, self, actions)
      self.status.update(realm, self, actions)
      self.history.update(realm, self, actions)

      if not self.alive:
         return False

      return True

   def receiveDamage(self, source, dmg):
      self.history.damage.update(dmg)
      self.resources.health.decrement(dmg)

      if not self.alive and source is not None:
         source.receiveLoot(self.loadout)
         return False

      return True

   def receiveLoot(self, loadout):
      pass

   def applyDamage(self, dmg, style):
      pass

   @property
   def pos(self):
      return self.base.pos

   @property
   def alive(self):
      if self.killed:
         return False

      if self.resources.health.empty:
         return False

      return True

   @property
   def isPlayer(self) -> bool:
      return False

   @property
   def isNPC(self) -> bool:
      return False
