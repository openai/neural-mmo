from neural_mmo.forge.blade.item import item, armor

class Inventory:
   def __init__(self, config):
      self.ammo = 0
      #self.resetArmor()
      #self.resetMelee()
      #self.resetRanged()

   def resetArmor(self):
      self.armor = Armor.Base()

   def resetMelee(self):
      self.melee = Item.Base()
      
   def resetRanged(self):
      self.ranged = Item.Base()
