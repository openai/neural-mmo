from pdb import set_trace as T
from nmmo.lib.colors import Tier

class Loadout:
   def __init__(self, chest=0, legs=0):
      self.chestplate = Chestplate(chest)
      self.platelegs  = Platelegs(legs) 

   @property
   def defense(self):
      return (self.chestplate.level + self.platelegs.level) // 2

   def packet(self):
      packet = {}

      packet['chestplate'] = self.chestplate.packet()
      packet['platelegs']  = self.platelegs.packet()

      return packet

class Armor:
   def __init__(self, level):
      self.level = level

   def packet(self):
     packet = {}
 
     packet['level'] = self.level
     packet['color'] = self.color.packet()

     return packet

   @property
   def color(self):
     if self.level == 0:
        return Tier.BLACK
     if self.level < 10:
        return Tier.WOOD
     elif self.level < 20:
        return Tier.BRONZE
     elif self.level < 40:
        return Tier.SILVER
     elif self.level < 60:
        return Tier.GOLD
     elif self.level < 80:
        return Tier.PLATINUM
     else:
        return Tier.DIAMOND
     

class Chestplate(Armor):
   pass

class Platelegs(Armor):
   pass

