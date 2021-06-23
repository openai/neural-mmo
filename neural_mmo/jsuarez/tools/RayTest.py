from pdb import set_trace as T
import numpy as np
import ray
import time

ray.init()

def timeit(f):
   def profile(*args, iters=1000):
      sumTime = 0
      for i in range(iters):
         start = time.time()
         f(*args) 
         sumTime += time.time() - start
      print(sumTime/iters)
   return profile

@timeit
def rayShared(x):
   #xId = ray.put(x)
   return ray.get(x)

@ray.remote
class Foo:
    def __init__(self):
        pass

    def bar(self):
        return 1

'''
foo = Foo.remote()
start = time.time()
for i in range(1000):
    a = foo.bar.remote()
print(time.time() - start)
'''


times = []
x = np.random.rand(15000, 15000)
x = ray.put(x)
#noise = [str(np.random.rand())*int(sz)]
rayShared(x, iters=1000)

'''
@ray.remote
class Foo(object):
   def method(self):
       return 1

a = Foo.remote()
start = time.time()
for i in range(1000):
   ray.get(a.method.remote())  # This also takes about 440us for me (on my laptop)
print(time.time()-start)
'''
