from pdb import set_trace as T
import numpy as np

import inspect

from nmmo.systems import item as Item
from nmmo.systems import skill as Skill

class Equipment:
   def __init__(self, realm):
      self.hat         = None
      self.top         = None
      self.bottom      = None

      self.held        = None
      self.ammunition  = None

      #Placeholder item for render
      self.itm         = Item.Hat(realm, 0)

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
   def packet(self):
      packet = {}

      self.conditional_packet(packet, 'hat',        self.hat)
      self.conditional_packet(packet, 'top',        self.top)
      self.conditional_packet(packet, 'bottom',     self.bottom)
      self.conditional_packet(packet, 'held',       self.held)
      self.conditional_packet(packet, 'ammunition', self.ammunition)

      packet['item_level']    = self.total(lambda e: e.level)

      packet['melee_attack']  = self.total(lambda e: e.melee_attack)
      packet['range_attack']  = self.total(lambda e: e.range_attack)
      packet['mage_attack']   = self.total(lambda e: e.mage_attack)
      packet['melee_defense'] = self.total(lambda e: e.melee_defense)
      packet['range_defense'] = self.total(lambda e: e.range_defense)
      packet['mage_defense']  = self.total(lambda e: e.mage_defense)

      return packet


class Inventory:
   def __init__(self, realm, entity):
      config           = realm.config
      self.realm       = realm
      self.entity      = entity
      self.config      = config

      self._items      = set()
      self.capacity    = config.INVENTORY_CAPACITY

      self.gold        = Item.Gold(realm)
      self.equipment   = Equipment(realm)

   @property
   def space(self):
      return self.capacity - len(self._items)

   @property
   def dataframeKeys(self):
      return [e.instanceID for e in self._items]

   def __contains__(self, item):
      if item in self._items:
         return True
      return False

   def packet(self):
      return {
            'items':     [self.gold.packet] + [e.packet for e in self._items],
            'equipment': self.equipment.packet}

   def __iter__(self):
      for item in self._items:
         yield item

   def receive(self, item):
      if __debug__:
         assert not item.equipped.val, 'Received equipped item {}'.format(item)
         assert self.space, 'Out of space for {}'.format(item)

      if not self.space:
         return

      if isinstance(item, Item.Gold):
         self.gold.quantity += item.quantity.val
         return

      if __debug__:
         assert item.quantity.val, 'Received empty item {}'.format(item)

      #TODO: reduce complexity
      if isinstance(item, Item.Ammunition):
         for itm in self._items:
            if type(itm) != type(item):
                continue
            if itm.level != item.level:
                continue
            itm.quantity += item.quantity.val
            return

      self._items.add(item)

   def remove(self, item, level=None):
      if __debug__:
         assert item in self._items, 'No item {} to remove'.format(item)

      if item.equipped.val:
          item.use(self.entity)

      if __debug__:
         assert not item.equipped.val, 'Removing item {} while equipped'.format(item)

      self._items.remove(item)
      return item
