from pdb import set_trace as T
from collections import defaultdict
import asyncio
import time
import ray
     
class Ascend:
   def __init__(self, config, idx):
      super().__init__()
      self.inbox = defaultdict(asyncio.Queue)
      self.idx   = idx

   async def put(self, packet, key):
      await self.inbox[key].put(packet)

   def recv(self, key):
      data = []
      while True:
         try:
            pkt = self.inbox[key].get_nowait()
            data.append(pkt)
         except asyncio.QueueEmpty:
            break
      return data

   @staticmethod
   def send(dests, packet, key):
      if type(dests) != list:
         dests = [dests]

      for dst in dests:
         try:
            dst.put.remote(packet, key)
         except:
            print('Error at {}'.format(dst))

@ray.remote
class Client(Ascend):
   def __init__(self):
      super().__init__(None, 0)
 
   async def run(self):
      print('Client: run')
      while True:
         await asyncio.sleep(0)
         time.sleep(1)
         pkt = self.recv('Server')
         print('Recv: {}'.format(pkt))

if __name__ == '__main__':
   ray.init()

   client = Client.remote()
   client.run.remote()

   while True:
      print('Sending packet')
      Ascend.send(client, 'Packet', 'Server')
      time.sleep(0.25)
      

