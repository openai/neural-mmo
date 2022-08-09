from pdb import set_trace as T

import logging
import random

from nmmo.io.stimulus import Serialized
from nmmo.lib.colors import Tier
from nmmo.systems import combat

class ItemID:
   item_ids = {} 
   id_items = {}

   def register(cls, item_id):
      if __debug__:
         if cls in ItemID.item_ids:
            assert ItemID.item_ids[cls] == item_id, f'Missmatched item_id assignment for class {cls}'
         if item_id in ItemID.id_items:
            assert ItemID.id_items[item_id] == cls, f'Missmatched class assignment for item_id {item_id}'

      ItemID.item_ids[cls] = item_id
      ItemID.id_items[item_id] = cls

   def get(cls_or_id):
      if type(cls_or_id) == int:
         return ItemID.id_items[cls_or_id]
      return ItemID.item_ids[cls_or_id]

class Item:
   ITEM_ID = None
   INSTANCE_ID = 0
   def __init__(self, realm, level,
         capacity=0, quantity=1, tradable=True,
         melee_attack=0, range_attack=0, mage_attack=0,
         melee_defense=0, range_defense=0, mage_defense=0,
         health_restore=0, resource_restore=0, price=0):

      self.config     = realm.config
      self.realm      = realm  

      self.instanceID = Item.INSTANCE_ID
      realm.items[self.instanceID] = self

      self.instance         = Serialized.Item.ID(realm.dataframe, self.instanceID, Item.INSTANCE_ID)
      self.index            = Serialized.Item.Index(realm.dataframe, self.instanceID, self.ITEM_ID)
      self.level            = Serialized.Item.Level(realm.dataframe, self.instanceID, level)
      self.capacity         = Serialized.Item.Capacity(realm.dataframe, self.instanceID, capacity)
      self.quantity         = Serialized.Item.Quantity(realm.dataframe, self.instanceID, quantity)
      self.tradable         = Serialized.Item.Tradable(realm.dataframe, self.instanceID, tradable)
      self.melee_attack     = Serialized.Item.MeleeAttack(realm.dataframe, self.instanceID, melee_attack)
      self.range_attack     = Serialized.Item.RangeAttack(realm.dataframe, self.instanceID, range_attack)
      self.mage_attack      = Serialized.Item.MageAttack(realm.dataframe, self.instanceID, mage_attack)
      self.melee_defense    = Serialized.Item.MeleeDefense(realm.dataframe, self.instanceID, melee_defense)
      self.range_defense    = Serialized.Item.RangeDefense(realm.dataframe, self.instanceID, range_defense)
      self.mage_defense     = Serialized.Item.MageDefense(realm.dataframe, self.instanceID, mage_defense)
      self.health_restore   = Serialized.Item.HealthRestore(realm.dataframe, self.instanceID, health_restore)
      self.resource_restore = Serialized.Item.ResourceRestore(realm.dataframe, self.instanceID, resource_restore)
      self.price            = Serialized.Item.Price(realm.dataframe, self.instanceID, price)
      self.equipped         = Serialized.Item.Equipped(realm.dataframe, self.instanceID, 0)

      realm.dataframe.init(Serialized.Item, self.instanceID, None)

      Item.INSTANCE_ID += 1
      if self.ITEM_ID is not None:
         ItemID.register(self.__class__, item_id=self.ITEM_ID)

   @property
   def signature(self):
      return (self.index.val, self.level.val)

   @property
   def packet(self):
      return {'item':             self.__class__.__name__,
              'level':            self.level.val,
              'capacity':         self.capacity.val,
              'quantity':         self.quantity.val,
              'melee_attack':     self.melee_attack.val,
              'range_attack':     self.range_attack.val,
              'mage_attack':      self.mage_attack.val,
              'melee_defense':    self.melee_defense.val,
              'range_defense':    self.range_defense.val,
              'mage_defense':     self.mage_defense.val,
              'health_restore':   self.health_restore.val,
              'resource_restore': self.resource_restore.val,
              'price':            self.price.val}
 
   def use(self, entity):
      return
      #TODO: Warning?
      #assert False, f'Use {type(self)} not defined'

class Stack():
    pass

class Gold(Item, Stack):
   ITEM_ID = 1
   def __init__(self, realm, **kwargs):
      super().__init__(realm, level=0, tradable=False, **kwargs)

class Equipment(Item):
   @property
   def packet(self):
     packet = {'color': self.color.packet()}
     return {**packet, **super().packet}

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

   def use(self, entity):
      if self.equipped.val:
         self.equipped.update(0)
         equip = self.unequip(entity)
      else:
         self.equipped.update(1)
         equip = self.equip(entity)

         config = self.config
         if not config.LOG_MILESTONES or not entity.isPlayer:
             return equip

         realm     = self.realm
         equipment = entity.equipment
         item_name = self.__class__.__name__

         if realm.quill.milestone.log_max(f'{item_name}_level', self.level.val) and config.LOG_VERBOSE:
            logging.info(f'EQUIPMENT: Equipped level {self.level.val} {item_name}')
         if realm.quill.milestone.log_max(f'Item_Level', equipment.item_level) and config.LOG_VERBOSE:
            logging.info(f'EQUIPMENT: Item level {equipment.item_level}')
         if realm.quill.milestone.log_max(f'Mage_Attack', equipment.mage_attack) and config.LOG_VERBOSE:
            logging.info(f'EQUIPMENT: Mage attack {equipment.mage_attack}')
         if realm.quill.milestone.log_max(f'Mage_Defense', equipment.mage_defense) and config.LOG_VERBOSE:
            logging.info(f'EQUIPMENT: Mage defense {equipment.mage_defense}')
         if realm.quill.milestone.log_max(f'Range_Attack', equipment.range_attack) and config.LOG_VERBOSE:
            logging.info(f'EQUIPMENT: Range attack {equipment.range_attack}')
         if realm.quill.milestone.log_max(f'Range_Defense', equipment.range_defense) and config.LOG_VERBOSE:
            logging.info(f'EQUIPMENT: Range defense {equipment.range_defense}')
         if realm.quill.milestone.log_max(f'Melee_Attack', equipment.melee_attack) and config.LOG_VERBOSE:
            logging.info(f'EQUIPMENT: Melee attack {equipment.melee_attack}')
         if realm.quill.milestone.log_max(f'Melee_Defense', equipment.melee_defense) and config.LOG_VERBOSE:
                logging.info(f'EQUIPMENT: Melee defense {equipment.melee_defense}')

      return equip
 
class Armor(Equipment):
   def __init__(self, realm, level, **kwargs):
      defense = realm.config.EQUIPMENT_ARMOR_BASE_DEFENSE + level*realm.config.EQUIPMENT_ARMOR_LEVEL_DEFENSE
      super().__init__(realm, level,
              melee_defense=defense,
              range_defense=defense,
              mage_defense=defense,
              **kwargs)

class Hat(Armor):
   ITEM_ID = 2

   def equip(self, entity):
      if entity.level < self.level.val:
          return
      if entity.inventory.equipment.hat:
          entity.inventory.equipment.hat.use(entity)
      entity.inventory.equipment.hat = self

   def unequip(self, entity):
      entity.inventory.equipment.hat = None

class Top(Armor):
   ITEM_ID = 3

   def equip(self, entity):
      if entity.level < self.level.val:
          return
      if entity.inventory.equipment.top:
          entity.inventory.equipment.top.use(entity)
      entity.inventory.equipment.top = self

   def unequip(self, entity):
      entity.inventory.equipment.top = None

class Bottom(Armor):
   ITEM_ID = 4

   def equip(self, entity):
      if entity.level < self.level.val:
          return
      if entity.inventory.equipment.bottom:
          entity.inventory.equipment.bottom.use(entity)
      entity.inventory.equipment.bottom = self

   def unequip(self, entity):
      entity.inventory.equipment.bottom = None

class Weapon(Equipment):
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, **kwargs)
      self.attack = realm.config.EQUIPMENT_WEAPON_BASE_DAMAGE + level*realm.config.EQUIPMENT_WEAPON_LEVEL_DAMAGE

   def equip(self, entity):
      if entity.inventory.equipment.held:
          entity.inventory.equipment.held.use(entity)
      entity.inventory.equipment.held = self

   def unequip(self, entity):
      entity.inventory.equipment.held = None

class Sword(Weapon):
   ITEM_ID = 5
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, **kwargs)
      self.melee_attack.update(self.attack)

   def equip(self, entity):
      if entity.skills.melee.level.val >= self.level.val:
         super().equip(entity)
 
class Bow(Weapon):
   ITEM_ID = 6
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, **kwargs)
      self.range_attack.update(self.attack)

   def equip(self, entity):
      if entity.skills.range.level.val >= self.level.val:
         super().equip(entity)
 
class Wand(Weapon):
   ITEM_ID = 7
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, **kwargs)
      self.mage_attack.update(self.attack)

   def equip(self, entity):
      if entity.skills.mage.level.val >= self.level.val:
         super().equip(entity)
 
class Tool(Equipment):
   def __init__(self, realm, level, **kwargs):
      defense = realm.config.EQUIPMENT_TOOL_BASE_DEFENSE + level*realm.config.EQUIPMENT_TOOL_LEVEL_DEFENSE
      super().__init__(realm, level,
              melee_defense=defense,
              range_defense=defense,
              mage_defense=defense,
              **kwargs)

   def equip(self, entity):
      if entity.inventory.equipment.held:
          entity.inventory.equipment.held.use(entity)
      entity.inventory.equipment.held = self

   def unequip(self, entity):
      entity.inventory.equipment.held = None

class Rod(Tool):
    ITEM_ID = 8
    def equip(self, entity):
       if entity.skills.fishing.level >= self.level.val:
          super().equip(entity)
          return True

       return False

class Gloves(Tool):
    ITEM_ID = 9
    def equip(self, entity):
       if entity.skills.herbalism.level >= self.level.val:
          super().equip(entity)
          return True

       return False

class Pickaxe(Tool):
    ITEM_ID = 10
    def equip(self, entity):
       if entity.skills.prospecting.level >= self.level.val:
          super().equip(entity)
          return True

       return False

class Chisel(Tool):
    ITEM_ID = 11
    def equip(self, entity):
       if entity.skills.carving.level >= self.level.val:
          super().equip(entity)
          return True

       return False

class Arcane(Tool):
    ITEM_ID = 12
    def equip(self, entity):
       if entity.skills.alchemy.level >= self.level.val:
          super().equip(entity)
          return True

       return False

class Ammunition(Equipment, Stack):
   def __init__(self, realm, level, **kwargs):
       super().__init__(realm, level, **kwargs)
       self.attack = realm.config.EQUIPMENT_AMMUNITION_BASE_DAMAGE + level*realm.config.EQUIPMENT_AMMUNITION_LEVEL_DAMAGE

   def equip(self, entity):
      if entity.inventory.equipment.ammunition:
          entity.inventory.equipment.ammunition.use(entity)
      entity.inventory.equipment.ammunition = self

   def unequip(self, entity):
      entity.inventory.equipment.ammunition = None

   def fire(self, entity):
      if __debug__:
         err = 'Used ammunition with 0 quantity'
         assert self.quantity.val > 0, err

      self.quantity.decrement()

      if self.quantity.val == 0:
         entity.inventory.remove(self)

      return self.damage
      
class Scrap(Ammunition):
   ITEM_ID = 13
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, **kwargs)
      self.melee_attack.update(self.attack)

   def equip(self, entity):
      if entity.skills.melee.level >= self.level.val:
          super().equip(entity)
          return True

      return False

   @property
   def damage(self):
      return self.melee_attack.val

class Shaving(Ammunition):
   ITEM_ID = 14
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, **kwargs)
      self.range_attack.update(self.attack)

   def equip(self, entity):
      if entity.skills.range.level >= self.level.val:
          super().equip(entity)
          return True

      return False

   @property
   def damage(self):
      return self.range_attack.val

class Shard(Ammunition):
   ITEM_ID = 15
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, **kwargs)
      self.mage_attack.update(self.attack)

   def equip(self, entity):
      if entity.skills.mage.level >= self.level.val:
          super().equip(entity)
          return True

      return False

   @property
   def damage(self):
      return self.mage_attack.val

class Consumable(Item):
    pass

class Ration(Consumable):
   ITEM_ID = 16
   def __init__(self, realm, level, **kwargs):
      restore = realm.config.PROFESSION_CONSUMABLE_RESTORE(level)
      super().__init__(realm, level, resource_restore=restore, **kwargs)

   def use(self, entity):
      if entity.level < self.level.val:
          return False

      if self.config.LOG_MILESTONES and self.realm.quill.milestone.log_max(f'Consumed_Ration', self.level.val) and self.config.LOG_VERBOSE:
         logging.info(f'PROFESSION: Consumed level {self.level.val} ration')

      entity.resources.food.increment(self.resource_restore.val)
      entity.resources.water.increment(self.resource_restore.val)

      entity.ration_level_consumed = max(entity.ration_level_consumed, self.level.val)
      entity.ration_consumed += 1

      entity.inventory.remove(self)

      return True

class Poultice(Consumable):
   ITEM_ID = 17

   def __init__(self, realm, level, **kwargs):
      restore = realm.config.PROFESSION_CONSUMABLE_RESTORE(level)
      super().__init__(realm, level, health_restore=restore, **kwargs)

   def use(self, entity):
      if entity.level < self.level.val:
          return False

      if self.config.LOG_MILESTONES and self.realm.quill.milestone.log_max(f'Consumed_Poultice', self.level.val) and self.config.LOG_VERBOSE:
         logging.info(f'PROFESSION: Consumed level {self.level.val} poultice')

      entity.resources.health.increment(self.health_restore.val)

      entity.poultice_level_consumed = max(entity.poultice_level_consumed, self.level.val)
      entity.poultice_consumed       += 1

      entity.inventory.remove(self)

      return True
