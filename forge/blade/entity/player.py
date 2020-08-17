import numpy as np
from pdb import set_trace as T

from forge.blade.systems import ai, equipment
from forge.blade.lib.enums import Material, Neon

from forge.blade.systems.skill import Skills
from forge.blade.systems.inventory import Inventory
from forge.blade.entity import entity

class Base(entity.Base):
   def __init__(self, config, pos, iden, pop, name, color):
      super().__init__(config, pos, iden, name, color)
      self.name = name + str(iden)
      self.population = pop
      self.self       = True

   def update(self, realm, entity, actions):
      super().update(realm, entity, actions)
      if entity.resources.health <= 0:
         self.killed = True
         return

   @property
   def pos(self):
      return self.r, self.c

   def packet(self):
      data = super().packet()

      data['population'] = self.population
      data['self']       = self.self

      return data

class Resources:
   def __init__(self, config):
      self.health = entity.Resource(config.HEALTH)
      self.water  = entity.Resource(config.RESOURCE)
      self.food   = entity.Resource(config.RESOURCE)

   def update(self, realm, entity, actions):
      self.health._max = entity.skills.constitution.level
      self.water._max  = entity.skills.fishing.level
      self.food._max   = entity.skills.hunting.level

   def packet(self):
      data = {}
      data['health'] = self.health.packet()
      data['food']   = self.food.packet()
      data['water']  = self.water.packet()
      return data

class Player(entity.Entity):
   SERIAL = 0
   def __init__(self, config, pos, iden, pop, name='', color=None):
      super().__init__(config, iden)
      self.annID  = pop
      self.target = None

      self.vision = 10
      self.food   = None
      self.water  = None

      #Submodules
      self.base      = Base(config, pos, iden, pop, name, color)
      self.status    = entity.Status(config)
      self.history   = entity.History(config)
      self.resources = Resources(config)
      self.skills    = Skills(config)
      self.loadout   = equipment.Loadout()
      #self.inventory = Inventory(config)
      #self.chat      = Chat(config)

   @property
   def alive(self):
      assert self.resources.health >= 0
      if self.resources.health == 0:
         return False
      return super().alive

   def applyDamage(self, dmg, style):
      self.resources.food.increment(dmg)
      self.resources.water.increment(dmg)

      self.skills.applyDamage(dmg, style)
      
   #Note: does not stack damage, but still applies to health
   def receiveDamage(self, source, dmg):
      self.history.damage = dmg

      if not self.alive:
         return 

      self.resources.health.decrement(dmg)
      self.resources.food.decrement(dmg)
      self.resources.water.decrement(dmg)

      self.skills.receiveDamage(dmg)

      if not self.alive and isinstance(source, Player):
         source.receiveLoot(self.loadout)
            
   def receiveLoot(self, loadout):
      if loadout.chestplate.level > self.loadout.chestplate.level:
         self.loadout.chestplate = loadout.chestplate
      if loadout.platelegs.level > self.loadout.platelegs.level:
         self.loadout.platelegs = loadout.platelegs

   @property
   def serial(self):
      return self.annID, self.entID
 
   def packet(self):
      data = super().packet()

      data['entID']    = self.entID
      data['annID']    = self.annID

      data['base']     = self.base.packet()
      data['resource'] = self.resources.packet()
      data['skills']   = self.skills.packet()

      return data
  
   #PCs interact with the world only through stimuli
   #to prevent cheating 
   def decide(self, packets):
      action, args = self.cpu.decide(self, packets)
      return action, args

   def step(self, realm, actions):
      self.base.update(realm, self, actions)
      if not self.alive:
         return

      self.resources.update(realm, self, actions)
      self.status.update(realm, self, actions)
      self.skills.update(realm, self, actions)
      self.history.update(realm, self, actions)
      #self.inventory.update(world, actions)
      #self.update(world, actions)

   def act(self, world, atnArgs):
      #Right now we only support one arg. So *args for future update
      atn, args = atnArgs
      args = args.values()
      atn.call(world, self, *args)


