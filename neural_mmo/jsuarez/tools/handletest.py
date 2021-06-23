import ray

ray.init(driver_mode=ray.PYTHON_MODE)

@ray.remote
class Actor1:
   def method(self):
       pass

@ray.remote
class Actor2:
   def __init__(self, a):
       self.a = a
   def method(self):
       ray.get(self.a.method.remote())

a1 = Actor1.remote()
a2 = Actor2.remote(a1)
ray.get(a2.method.remote())
