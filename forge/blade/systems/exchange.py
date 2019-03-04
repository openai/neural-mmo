from collections import defaultdict
from queue import PriorityQueue

class Offer:
   def __init__(self, item, quant, itemPrice, coffer=0):
      self.item = item
      self.quantLeft = quant
      self.quantFulfilled = 0
      self.itemPrice = itemPrice
      self.coffer = coffer

   @property
   def complete(self):
      return self.quantLeft == 0

   def partialCollect(self):
      pass

   def cancel(self):
      return self.coffer, [self.item() for e in self.quantFulfilled]

   def __lt__(self, other):
      return True

   def __eq__(self, other):
      return False

class BuyOffer(Offer):
   def buy(self, quant, itemPrice):
      self.coffer -= itemPrice * quant

   #Collect only profits thus far
   def partialCollect(self):
      ret = [self.item() for e in self.quantFulfilled]
      self.quantFulfilled = 0
      return ret

class SellOffer(Offer):
   def sell(self, quant):
      self.coffer += self.itemPrice * quant
      self.quantLeft -= quant
      assert self.quantLeft >= 0

   def partialCollect(self):
      ret = self.coffer
      self.coffer = 0
      return self.coffer

#Why is there no peek fuction...
class PQ(PriorityQueue):
   def peek(self):
      if len(self.queue) > 0:
         return self.queue[0]
      return None

class Exchange:
   def __init__(self):
      self.buyOffers  = defaultdict(PQ)
      self.sellOffers = defaultdict(PQ)

   def buy(self, item, quant, maxPrice):
      offer = BuyOffer(item, quant, maxPrice, coffer=quant*maxPrice)
      self.buyOffers[item].put(offer, -maxPrice)
      self.update(item)
      return offer

   def sell(self, item, quant, itemPrice):
      offer = SellOffer(item, quant, itemPrice)
      self.sellOffers[item].put(offer, itemPrice)
      self.update(item)
      return offer

   def update(self, item):
      buyOffer  = self.buyOffers[item].peek()
      sellOffer = self.sellOffers[item].peek()

      if None in (buyOffer, sellOffer):
         return
      maxBuy, minSell = buyOffer.itemPrice, sellOffer.itemPrice
      itemPrice = minSell #Advantage given to buyer arbitrarily

      if maxBuy >= minSell:
         if sellOffer.quantLeft < buyOffer.quantLeft:
            buyOffer.buy(sellOffer.quantLeft, itemPrice)
            sellOffer.sell(sellOffer.quantLeft)
            self.sellOffers[item].get()
         elif sellOffer.quantLeft > buyOffer.quantLeft:
            buyOffer.buy(buyOffer.quantLeft, itemPrice)
            sellOffer.sell(buyOffer.quantLeft)
            self.buyOffers[item].get()
         elif sellOffer.quantLeft == buyOffer.quantLeft:
            buyOffers.buy(buyOffer.quantLeft, itemPrice)
            sellOffer.sell(sellOffer.quantLeft)
            self.buyOffers[item].get()
            self.sellOffers[item].get()

         self.update(item)
