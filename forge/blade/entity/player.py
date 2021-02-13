import numpy as np
from pdb import set_trace as T

from forge.blade.systems import ai, equipment
from forge.blade.lib import material

from forge.blade.systems.skill import Skills
from forge.blade.systems.inventory import Inventory
from forge.blade.entity import entity
from forge.blade.io.stimulus import Static

class Player(entity.Entity):
   def __init__(self, realm, pos, iden, pop, name='', color=None):
      super().__init__(realm, pos, iden, name, color, pop)
      self.pop    = pop

      #Scripted hooks
      self.target = None
      self.food   = None
      self.water  = None
      self.vision = 7

      #Submodules
      self.skills     = Skills(self)
      #self.inventory = Inventory(dataframe)
      #self.chat      = Chat(dataframe)

      #Update immune
      mmul = realm.config.IMMUNE_MUL
      madd = realm.config.IMMUNE_ADD
      mmax = realm.config.IMMUNE_MAX
      immune = min(mmul*len(realm.players) + madd, mmax)
      self.status.immune.update(immune)

      self.dataframe.init(Static.Entity, self.entID, self.pos)

   @property
   def serial(self):
      return self.population, self.entID

   @property
   def isPlayer(self) -> bool:
      return True

   @property
   def population(self):
      return self.pop

   def applyDamage(self, dmg, style):
      self.resources.food.increment(dmg)
      self.resources.water.increment(dmg)
      self.skills.applyDamage(dmg, style)
      
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

   def packet(self):
      data = super().packet()

      data['entID']    = self.entID
      data['annID']    = self.population

      data['base']     = self.base.packet()
      data['resource'] = self.resources.packet()
      data['skills']   = self.skills.packet()

      return data
  
   def update(self, realm, actions):
      '''Post-action update. Do not include history'''
      super().update(realm, actions)

      if not self.alive:
         return

      self.resources.update(realm, self, actions)
      self.skills.update(realm, self, actions)
      #self.inventory.update(world, actions)
