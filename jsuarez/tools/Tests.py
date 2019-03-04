from pdb import set_trace as T
from sim import Exchange
from sim.item import RawFish

def testExchange():
   market = Exchange.Exchange()
   sardine = RawFish.Sardine

   def checkPrint():
      print(market.buyOffers[sardine].queue)
      print(market.sellOffers[sardine].queue)
      print()

   market.buy(sardine, 1, 90)
   checkPrint()
   market.sell(sardine, 2, 100)
   checkPrint()
   market.buy(sardine, 3, 130)
   checkPrint()
   market.sell(sardine, 5, 90)
   checkPrint()
   market.buy(sardine, 5, 100)
   checkPrint()

if __name__ == '__main__':
   testExchange()
