import numpy as np
from pdb import set_trace as T

from forge.blade.systems import ai, equipment
from forge.blade.lib.enums import Material, Neon

from forge.blade.systems.skill import Skills
from forge.blade.systems.inventory import Inventory
from forge.blade.entity import entity
from forge.blade.io.stimulus import Static

class Player(entity.Entity):
   def __init__(self, realm, pos, iden, pop, name='', color=None):
      super().__init__(realm, pos, iden, name, color, pop)

      self.annID  = pop
      self.target = None

      self.vision = 7

      #Submodules
      self.skills     = Skills(self)
      #self.inventory = Inventory(dataframe)
      #self.chat      = Chat(dataframe)

      self.dataframe.init(Static.Entity, self.entID, self.pos)

   def applyDamage(self, dmg, style):
      self.resources.food.increment(dmg)
      self.resources.water.increment(dmg)
      self.skills.applyDamage(dmg, style)
      
   #Note: does not stack damage, but still applies to health
   def receiveDamage(self, source, dmg):
      if not super().receiveDamage(source, dmg):
         return 

      self.resources.food.decrement(dmg)
      self.resources.water.decrement(dmg)
      self.skills.receiveDamage(dmg)

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

   def update(self, realm, actions):
      '''Post-action update. Do not include history'''
      if not super().update(realm, actions):
         return

      self.resources.update(realm, self, actions)
      self.skills.update(realm, self, actions)
      #self.inventory.update(world, actions)

   def act(self, world, atnArgs):
      #Right now we only support one arg. So *args for future update
      atn, args = atnArgs
      args = args.values()
      atn.call(world, self, *args)


