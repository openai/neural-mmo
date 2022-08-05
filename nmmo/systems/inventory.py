from pdb import set_trace as T
import numpy as np

import inspect
import logging

from nmmo.systems import item as Item
from nmmo.systems import skill as Skill

class Equipment:
   def __init__(self, realm):
      self.hat         = None
      self.top         = None
      self.bottom      = None

      self.held        = None
      self.ammunition  = None

   def total(self, lambda_getter):
      items = [lambda_getter(e).val for e in self]
      if not items:
          return 0
      return sum(items)

   def __iter__(self):
      for item in [self.hat, self.top, self.bottom, self.held, self.ammunition]:
         if item is not None:
            yield item

   def conditional_packet(self, packet, item_name, item):
      if item:
         packet[item_name] = item.packet

   @property
   def item_level(self):
       return self.total(lambda e: e.level)

   @property
   def melee_attack(self):
       return self.total(lambda e: e.melee_attack)

   @property
   def range_attack(self):
       return self.total(lambda e: e.range_attack)

   @property
   def mage_attack(self):
       return self.total(lambda e: e.mage_attack)

   @property
   def melee_defense(self):
       return self.total(lambda e: e.melee_defense)

   @property
   def range_defense(self):
       return self.total(lambda e: e.range_defense)

   @property
   def mage_defense(self):
       return self.total(lambda e: e.mage_defense)

   @property
   def packet(self):
      packet = {}

      self.conditional_packet(packet, 'hat',        self.hat)
      self.conditional_packet(packet, 'top',        self.top)
      self.conditional_packet(packet, 'bottom',     self.bottom)
      self.conditional_packet(packet, 'held',       self.held)
      self.conditional_packet(packet, 'ammunition', self.ammunition)

      packet['item_level']    = self.item_level

      packet['melee_attack']  = self.melee_attack
      packet['range_attack']  = self.range_attack
      packet['mage_attack']   = self.mage_attack
      packet['melee_defense'] = self.melee_defense
      packet['range_defense'] = self.range_defense
      packet['mage_defense']  = self.mage_defense

      return packet


class Inventory:
   def __init__(self, realm, entity):
      config           = realm.config
      self.realm       = realm
      self.entity      = entity
      self.config      = config

      self.equipment   = Equipment(realm)

      if not config.ITEM_SYSTEM_ENABLED:
          return

      self.capacity         = config.ITEM_INVENTORY_CAPACITY
      self.gold             = Item.Gold(realm)

      self._item_stacks     = {self.gold.signature: self.gold}
      self._item_references = {self.gold}

   @property
   def space(self):
      return self.capacity - len(self._item_references)

   @property
   def dataframeKeys(self):
      return [e.instanceID for e in self._item_references]

   def packet(self):
      item_packet = []
      if self.config.ITEM_SYSTEM_ENABLED:
          item_packet = [e.packet for e in self._item_references]

      return {
            'items':     item_packet,
            'equipment': self.equipment.packet}

   def __iter__(self):
      for item in self._item_references:
         yield item

   def receive(self, item):
      assert isinstance(item, Item.Item), f'{item} received is not an Item instance'
      assert item not in self._item_references, f'{item} object received already in inventory'
      assert not item.equipped.val, f'Received equipped item {item}'
      assert self.space, f'Out of space for {item}'
      assert item.quantity.val, f'Received empty item {item}'

      config = self.config
      if config.LOG_MILESTONES and self.realm.quill.milestone.log_max(f'Receive_{item.__class__.__name__}', item.level.val) and config.LOG_VERBOSE:
          logging.info(f'INVENTORY: Received level {item.level.val} {item.__class__.__name__}')

      if isinstance(item, Item.Stack):
          signature = item.signature
          if signature in self._item_stacks:
              stack = self._item_stacks[signature]
              assert item.level.val == stack.level.val, f'{item} stack level mismatch'
              stack.quantity += item.quantity.val

              if config.LOG_MILESTONES and isinstance(item, Item.Gold) and self.realm.quill.milestone.log_max(f'Wealth', self.gold.quantity.val) and config.LOG_VERBOSE:
                  logging.info(f'EXCHANGE: Total wealth {self.gold.quantity.val} gold')
              
              return

          self._item_stacks[signature] = item

      self._item_references.add(item)

   def remove(self, item, quantity=None):
      assert isinstance(item, Item.Item), f'{item} received is not an Item instance'
      assert item in self._item_references, f'No item {item} to remove'

      if item.equipped.val:
          item.use(self.entity)

      assert not item.equipped.val, f'Removing {item} while equipped'

      if isinstance(item, Item.Stack):
         signature = item.signature 

         assert item.signature in self._item_stacks, f'{item} stack to remove not in inventory'
         stack = self._item_stacks[signature]

         if quantity is None or stack.quantity.val == quantity:
            self._item_references.remove(stack)
            del self._item_stacks[signature]
            return

         assert 0 < quantity <= stack.quantity.val, f'Invalid remove {quantity} x {item} ({stack.quantity.val} available)'
         stack.quantity.val -= quantity

         return

      self._item_references.remove(item)
