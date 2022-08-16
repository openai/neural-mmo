from pdb import set_trace as T
import numpy as np

import nmmo
from nmmo.systems import skill, droptable, combat, equipment, inventory
from nmmo.lib import material, utils

class Resources:
   def __init__(self, ent):
      self.health = nmmo.Serialized.Entity.Health(ent.dataframe, ent.entID)
      self.water  = nmmo.Serialized.Entity.Water( ent.dataframe, ent.entID)
      self.food   = nmmo.Serialized.Entity.Food(  ent.dataframe, ent.entID)

   def update(self, realm, entity, actions):
      config = realm.config

      if not config.RESOURCE_SYSTEM_ENABLED:
         return

      self.water.max = config.RESOURCE_BASE
      self.food.max  = config.RESOURCE_BASE

      regen  = config.RESOURCE_HEALTH_RESTORE_FRACTION
      thresh = config.RESOURCE_HEALTH_REGEN_THRESHOLD

      food_thresh  = self.food  > thresh * config.RESOURCE_BASE
      water_thresh = self.water > thresh * config.RESOURCE_BASE

      if food_thresh and water_thresh:
          restore = np.floor(self.health.max * regen)
          self.health.increment(restore)

      if self.food.empty:
          self.health.decrement(config.RESOURCE_STARVATION_RATE)

      if self.water.empty:
          self.health.decrement(config.RESOURCE_DEHYDRATION_RATE)

   def packet(self):
      data = {}
      data['health'] = self.health.packet()
      data['food']   = self.food.packet()
      data['water']  = self.water.packet()
      return data

class Status:
   def __init__(self, ent):
      self.config = ent.config
      self.freeze = nmmo.Serialized.Entity.Freeze(ent.dataframe, ent.entID)

   def update(self, realm, entity, actions):
      self.freeze.decrement()

   def packet(self):
      data = {}
      data['freeze'] = self.freeze.val
      return data

class History:
   def __init__(self, ent):
      self.actions = {}
      self.attack  = None
  
      self.origPos     = ent.pos
      self.exploration = 0
      self.playerKills = 0

      self.damage_received = 0
      self.damage_inflicted = 0

      self.damage    = nmmo.Serialized.Entity.Damage(   ent.dataframe, ent.entID)
      self.timeAlive = nmmo.Serialized.Entity.TimeAlive(ent.dataframe, ent.entID)

      self.lastPos = None

   def update(self, realm, entity, actions):
      self.attack  = None
      self.damage.update(0)

      self.actions = {}
      if entity.entID in actions:
          self.actions = actions[entity.entID]
 
      exploration      = utils.linf(entity.pos, self.origPos)
      self.exploration = max(exploration, self.exploration)

      self.timeAlive.increment()

   def packet(self):
      data = {}
      data['damage']    = self.damage.val
      data['timeAlive'] = self.timeAlive.val
      data['damage_inflicted'] = self.damage_inflicted
      data['damage_received'] = self.damage_received

      if self.attack is not None:
         data['attack'] = self.attack

      actions = {}
      for atn, args in self.actions.items():
         atn_packet = {}

         #Avoid recursive player packet
         if atn.__name__ == 'Attack':
             continue

         for key, val in args.items():
            if hasattr(val, 'packet'):
               atn_packet[key.__name__] = val.packet
            else:
               atn_packet[key.__name__] = val.__name__
         actions[atn.__name__] = atn_packet
      data['actions'] = actions

      return data

class Base:
   def __init__(self, ent, pos, iden, name, color, pop):
      self.name  = name + str(iden)
      self.color = color
      r, c       = pos

      self.r          = nmmo.Serialized.Entity.R(ent.dataframe, ent.entID, r)
      self.c          = nmmo.Serialized.Entity.C(ent.dataframe, ent.entID, c)

      self.population = nmmo.Serialized.Entity.Population(ent.dataframe, ent.entID, pop)
      self.self       = nmmo.Serialized.Entity.Self(      ent.dataframe, ent.entID, 1)
      self.identity   = nmmo.Serialized.Entity.ID(        ent.dataframe, ent.entID, ent.entID)
      self.level      = nmmo.Serialized.Entity.Level(     ent.dataframe, ent.entID, 3)
      self.item_level = nmmo.Serialized.Entity.ItemLevel( ent.dataframe, ent.entID, 0)
      self.gold       = nmmo.Serialized.Entity.Gold(      ent.dataframe, ent.entID, 0)
      self.comm       = nmmo.Serialized.Entity.Comm(      ent.dataframe, ent.entID, 0)

      ent.dataframe.init(nmmo.Serialized.Entity, ent.entID, (r, c))

   def update(self, realm, entity, actions):
      self.level.update(combat.level(entity.skills))

      if realm.config.EQUIPMENT_SYSTEM_ENABLED:
         self.item_level.update(entity.equipment.total(lambda e: e.level))

      if realm.config.EXCHANGE_SYSTEM_ENABLED:
         self.gold.update(entity.inventory.gold.quantity.val)

   @property
   def pos(self):
      return self.r.val, self.c.val

   def packet(self):
      data = {}

      data['r']          = self.r.val
      data['c']          = self.c.val
      data['name']       = self.name
      data['level']      = self.level.val
      data['item_level'] = self.item_level.val
      data['color']      = self.color.packet()
      data['population'] = self.population.val
      data['self']       = self.self.val

      return data

class Entity:
   def __init__(self, realm, pos, iden, name, color, pop):
      self.realm        = realm
      self.dataframe    = realm.dataframe
      self.config       = realm.config

      self.policy       = name
      self.entID        = iden
      self.repr         = None
      self.vision       = 5

      self.attacker     = None
      self.target       = None
      self.closest      = None
      self.spawnPos     = pos

      self.attackerID = nmmo.Serialized.Entity.AttackerID(self.dataframe, self.entID, 0)

      #Submodules
      self.base      = Base(self, pos, iden, name, color, pop)
      self.status    = Status(self)
      self.history   = History(self)
      self.resources = Resources(self)

      self.inventory = inventory.Inventory(realm, self)

   def packet(self):
      data = {}

      data['status']    = self.status.packet()
      data['history']   = self.history.packet()
      data['inventory'] = self.inventory.packet()
      data['alive']     = self.alive

      return data

   def update(self, realm, actions):
      '''Update occurs after actions, e.g. does not include history'''
      if self.history.damage == 0:
         self.attacker = None
         self.attackerID.update(0)

      self.base.update(realm, self, actions)
      self.status.update(realm, self, actions)
      self.history.update(realm, self, actions)

   def receiveDamage(self, source, dmg):
      self.history.damage_received += dmg
      self.history.damage.update(dmg)
      self.resources.health.decrement(dmg)

      if self.alive:
          return True

      if source is None:
          return True 

      if not source.isPlayer:
          return True 

      return False

   def applyDamage(self, dmg, style):
      self.history.damage_inflicted += dmg

   @property
   def pos(self):
      return self.base.pos

   @property
   def alive(self):
      if self.resources.health.empty:
         return False

      return True

   @property
   def isPlayer(self) -> bool:
      return False

   @property
   def isNPC(self) -> bool:
      return False

   @property
   def level(self) -> int:
       melee  = self.skills.melee.level.val
       ranged = self.skills.range.level.val
       mage   = self.skills.mage.level.val

       return int(max(melee, ranged, mage))
