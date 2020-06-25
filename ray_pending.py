import time
import ray

@ray.remote
class A:
   def __init__(self, b):
      self.b = b

   def thunk(self):
      self.b.thunk.remote()

@ray.remote
class B:
   def thunk(self):
      time.sleep(999999999)
   

ray.init()
b = B.remote()
a = A.remote(b)
while True:
   time.sleep(0.02)
   a.thunk.remote()
