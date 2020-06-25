from pdb import set_trace as T
from collections import defaultdict
import asyncio
import time
import ray
     
@ray.remote
class AsyncQueue:
   def __init__(self):
      self.inbox = defaultdict(asyncio.Queue)

   async def put(self, packet, key):
      print('Put data')
      await self.inbox[key].put(packet)

   async def get(self, key):
      data = []
      while True:
         try:
            pkt = self.inbox[key].get_nowait()
            data.append(pkt)
         except asyncio.QueueEmpty:
            break
      return data

class Ascend:
   def __init__(self, config, idx, queue=None):
      super().__init__()
      self.queue = queue
      self.idx   = idx

   def recv(self, key):
      return ray.get(self.queue.get.remote(key))

   @staticmethod
   def send(dests, packet, key):
      if type(dests) != list:
         dests = [dests]

      for dst in dests:
         try:
            dst.put.remote(packet, key)
         except Exception as e:
            print('Error at {}: {}'.format(dst, e))

@ray.remote
class Client(Ascend):
   def __init__(self, queue):
      super().__init__(None, 0, queue)
 
   def run(self):
      print('Client: run')
      while True:
         time.sleep(1)
         print('Tick')
         pkt = self.recv('Server')
         print('Recv: {}'.format(pkt))

if __name__ == '__main__':
   ray.init()

   queue  = AsyncQueue.remote()
   client = Client.remote(queue)
   client.run.remote()

   while True:
      print('Sending packet')
      Ascend.send(queue, 'Packet', 'Server')
      time.sleep(0.25)
      

