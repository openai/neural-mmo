from neural_mmo.forge.blade.item import item, armor

class Equipment:
   def __init__(self):
      self.ammo = 0
      self.resetArmor()
      self.resetMelee()
      self.resetRanged()

   def resetArmor(self):
      self.armor = armor.Base()

   def resetMelee(self):
      self.melee = item.Base()
      
   def resetRanged(self):
      self.ranged = item.Base()

