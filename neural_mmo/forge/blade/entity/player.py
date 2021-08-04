import numpy as np
from pdb import set_trace as T

from neural_mmo.forge.blade.systems import ai, equipment
from neural_mmo.forge.blade.lib import material

from neural_mmo.forge.blade.systems.skill import Skills
from neural_mmo.forge.blade.systems.achievement import Diary
from neural_mmo.forge.blade.entity import entity
from neural_mmo.forge.blade.io.stimulus import Static

class Player(entity.Entity):
   def __init__(self, realm, pos, agent):
      super().__init__(realm, pos, agent.iden, agent.name, agent.color, agent.pop)
      self.agent  = agent
      self.pop    = agent.pop

      #Scripted hooks
      self.target = None
      self.food   = None
      self.water  = None
      self.vision = 7

      #Submodules
      self.skills       = Skills(self)
      self.achievements = Diary(realm.config)

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
         if source:
            source.history.playerKills += 1
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
      self.achievements.update(realm, self)
      #self.inventory.update(world, actions)
