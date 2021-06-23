from collections import deque

import numpy as np
from queue import PriorityQueue
from neural_mmo.forge.blade.systems import combat

class ExchangeEntry:
   def __init__(self, item, numBuy, numSell, maxBuyPrice, maxSellPrice):
      self.item = item
      self.numBuy = numBuy
      self.numSell = numSell
      self.maxBuyPrice = maxBuyPrice
      self.maxSellPrice = maxSellPrice

   #Hackey str get. Only for printing
   @property
   def itemName(self):
      return str(self.item)[17:-2]

   @property
   def worth(self):
      return (self.numBuy + self.numSell)*self.maxBuyPrice

   def __lt__(self, other):
      return True

   def __eq__(self, other):
      return False

class StatTraker:
   def __init__(self, maxLen=2048):
      self.lenTrak = deque(maxlen=maxLen)

   def update(self, pcs, npcs, exchange):
      self.pcs = pcs
      self.npcs = npcs
      
      self.lenTrak.append(len(pcs))
      #Update exchange
      self.updateExchange(exchange)
   
   def updateExchange(self, exchange):
      self.exchange = PriorityQueue()
      buyOffers = exchange.buyOffers
      sellOffers = exchange.sellOffers
      buyKeys, sellKeys = buyOffers.keys(), sellOffers.keys()
      exchangeKeys = list(set(list(buyKeys) + list(sellKeys)))

      for key in exchangeKeys:
         keyBuys, keySells  = buyOffers[key], sellOffers[key]
         topBuy  = keyBuys.peek()
         topSell = keySells.peek()

         if topBuy is not None:
            numBuy, maxBuyPrice   = topBuy.quantLeft, topBuy.itemPrice
            item = topBuy.item
         else:
            numBuy, maxBuyPrice = 0, 0

         if topSell is not None:
            numSell, minSellPrice = topSell.quantLeft, topSell.itemPrice
            item = topSell.item
         else:
            numSell, minSellPrice = 0, 0

         #Compute total buy value
         totalBuy = 0
         for e in keyBuys.queue:
            totalBuy += e.coffer

         entry = ExchangeEntry(item, numBuy, numSell, 
               maxBuyPrice, minSellPrice)

         self.exchange.put(entry, totalBuy)
 
   @property
   def numEntities(self):
      return np.asarray(list(self.lenTrak))

class StatBlock:
   def __init__(self, entity):
      self.timeAlive = entity.timeAlive
      self.level   = combat.combatLevel(entity.skills)
      self.defense = entity.skills.defense.level
      self.melee   = entity.skills.melee.level
      self.ranged  = entity.skills.ranged.level
      
   
