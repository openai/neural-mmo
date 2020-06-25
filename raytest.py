from pdb import set_trace as T
import numpy as np
import ray
import time

@ray.remote
def work():
   return

ray.init()
t = time.time()

n, cores = 50, 6
for i in range(n):
   ret = []
   for j in range(cores):
      ret.append(work.remote())
   ray.get(ret)

print('Time: ', (time.time() - t)/n)

