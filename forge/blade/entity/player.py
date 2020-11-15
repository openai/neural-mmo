import numpy as np
from pdb import set_trace as T

from forge.blade.systems import ai, equipment
from forge.blade.lib.enums import Material, Neon

from forge.blade.systems.skill import Skills
from forge.blade.systems.inventory import Inventory
from forge.blade.entity import entity
from forge.blade.io.stimulus import Static

class Base(entity.Base):
   def __init__(self, ent, pos, iden, pop, name, color):
      super().__init__(ent, pos, iden, name, color)
      self.name = name + str(iden)

      self.population = Static.Entity.Population(ent.dataframe, ent.entID, pop)
      ent.dataframe.init(Static.Entity, ent.entID, (self.r.val, self.c.val))

   def update(self, realm, entity, actions):
      super().update(realm, entity, actions)
      if entity.resources.health <= 0:
         self.killed = True
         return

   def packet(self):
      data = super().packet()

      data['population'] = self.population.val
      data['self']       = self.self.val

      return data

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

class Player(entity.Entity):
   SERIAL = 0
   def __init__(self, realm, pos, iden, pop, name='', color=None):
      super().__init__(realm, iden)

      self.annID  = pop
      self.target = None

      self.vision = 7
      self.food   = None
      self.water  = None

      #Submodules
      self.base      = Base(self, pos, iden, pop, name, color)
      self.status    = entity.Status(self)
      self.history   = entity.History(self)
      self.resources = Resources(self)
      self.skills    = Skills(self)
      self.loadout   = equipment.Loadout()
      #self.inventory = Inventory(dataframe)
      #self.chat      = Chat(dataframe)

      self.dataframe.init(Static.Entity, self.entID, self.pos)

   @property
   def alive(self):
      #Have to change comparisons for these. Using str value in stim node.
      assert self.resources.health.val >= 0
      if self.resources.health.val == 0:
         return False
      return super().alive

   def applyDamage(self, dmg, style):
      self.resources.food.increment(dmg)
      self.resources.water.increment(dmg)

      self.skills.applyDamage(dmg, style)
      
   #Note: does not stack damage, but still applies to health
   def receiveDamage(self, source, dmg):
      self.history.damage.update(dmg)

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


