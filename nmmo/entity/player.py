import numpy as np
from pdb import set_trace as T

import nmmo
from nmmo.systems import ai, equipment
from nmmo.lib import material

from nmmo.systems.skill import Skills
from nmmo.systems.achievement import Diary
from nmmo.entity import entity

class Player(entity.Entity):
   def __init__(self, realm, pos, agent, color, pop):
      super().__init__(realm, pos, agent.iden, agent.name, color, pop)

      self.agent  = agent
      self.pop    = pop

      #Scripted hooks
      self.target = None
      self.food   = None
      self.water  = None
      self.vision = 7

      #Submodules
      self.skills = Skills(self)

      self.diary  = None
      if tasks := realm.config.TASKS:
          self.diary = Diary(tasks)

      self.dataframe.init(nmmo.Serialized.Entity, self.entID, self.pos)

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

      if self.diary:
         self.diary.update(realm, self)
