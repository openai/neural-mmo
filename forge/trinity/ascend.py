from pdb import set_trace as T
import ray, time

class Timed:
   def __init__(self):
      self.run_time  = 0
      self.wait_time = 0

   @property
   def time(self):
      run = self.run_time
      self.run_time = 0

      wait = self.wait_time
      self.wait_time = 0

      return run, wait

   @property
   def name(self):
      return self.__class__.__name__

   def logs(self):
      run, wait = self.time
      ret = Log(self.name, run, wait)
      return ret

class Log:
   def __init__(self, cls, runTime, waitTime):
      self.cls  = cls
      self.run  = runTime
      self.wait = waitTime
      self.disciples = []

def waittime(func):
   def decorated(self, *args):
      t = time.time()
      ret = func(self, *args)
      t = time.time() - t
      self.wait_time += t
      return ret

   return decorated

def runtime(func):
   def decorated(self, *args):
      t = time.time()
      ret = func(self, *args)
      t = time.time() - t
      self.run_time += t
      return ret

   return decorated


class Ascend(Timed):
   '''A featherweight ray wrapper for persistent,
   multilevel, synchronous and asynchronous computation 
   
   Args:
      disciple: a class to instantiate remotely
      n: number of remote instances
      *args: arguments for the disciple
   '''
   def __init__(self, disciple, n, *args):
      super().__init__()
      self.remote    = Ascend.isRemote(disciple)
      disciple       = Ascend.localize(disciple, self.remote)
      self.disciples = [disciple(*args, idx) for idx in range(n)]

   def distrib(self, *args):
      '''Asynchronous wrapper around the step function
      function of all remote disciples

      Args:
         *args: Arbitrary data to broadcast to disciples
         
      Returns:
         A list of async handles to the step returns
         from all remote disciples
      '''
      rets = []
      for disciple in self.disciples:
         step = Ascend.localize(disciple.step, self.remote)
         rets.append(step(*args))
      return rets

   @waittime
   def sync(self, rets):
      '''Synchronizes returns from distrib

      Args:
         rets: async handles returned from distrib

      Returns:
         A list of returns from all disciples
      '''
      if self.remote:
         return ray.get(rets)
      return rets

   def step(self, *args):
      '''Synchronous wrapper around the step function
      function of all remote disciples

      Args:
         *args: broadcast to all disciples
 
      Returns:
         A list of returns from all disciples
      '''
      rets = self.distrib(*args)
      return self.sync(rets)
  
   def localize(f, remote):
      return f if not remote else f.remote

   def isRemote(obj):
      return hasattr(obj, 'remote') or hasattr(obj, '__ray_checkpoint__')


