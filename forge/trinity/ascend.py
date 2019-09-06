from pdb import set_trace as T
import ray, time

from collections import defaultdict

class Timed:
   '''Performance logging superclass'''
   def __init__(self):
      self.run_time  = 0
      self.wait_time = 0

   @property
   def time(self):
      run  = self.run_time
      wait = self.wait_time
      return run, wait

   def resetLogs(self):
      self.run_time  = 0
      self.wait_time = 0

   @property
   def name(self):
      return self.__class__.__name__

   def logs(self):
      run, wait = self.time
      self.resetLogs()
      ret = {self.name: Log(run, wait)}
      return ret

class Log:
   def __init__(self, runTime, waitTime):
      self.run  = runTime
      self.wait = waitTime

   def merge(logs):
      run  = max([log.run for log in logs])
      wait = max([log.wait for log in logs])
      return Log(run, wait)

   def summary(logs):
      data = defaultdict(list)

      for log in logs:
         for key, val in log.items():
            data[key].append(val)         
         
      for key, logList in data.items():
         data[key] = Log.merge(logList)

      return data

   def aggregate(log):
      ret = defaultdict(dict)
      for key, val in log.items():
         ret['run'][key]  = val.run - val.wait
         ret['wait'][key] = val.wait
      return ret

def waittime(func):
   '''Performance profiling decorator'''
   def decorated(self, *args):
      t = time.time()
      ret = func(self, *args)
      t = time.time() - t
      self.wait_time += t
      return ret

   return decorated

def runtime(func):
   '''Performance profiling decorator'''
   def decorated(self, *args):
      t = time.time()
      ret = func(self, *args)
      t = time.time() - t
      self.run_time += t
      return ret

   return decorated

class Ascend(Timed):
   '''A featherweight Ray wrapper for persistent,
   multilevel, synchronous and asynchronous computation 

   Provides:
      - A synchronous step() function for performing
      persistent computation across remote workers
      - An asynchronous distrib() function that
      steps remote workers without waiting for returns
      - A sync() function for collecting distrib() returns
      - Various smaller tools + Timed logging

   Works with pdb while using Ray local mode

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

   def distrib(self, *args, shard=None):
      '''Asynchronous wrapper around the step function
      function of all remote disciples

      Args:
         *args: Arbitrary data to broadcast to disciples
         shard: A binary mask of length len(args) specifying whether to scatter each argument
         
      Returns:
         A list of async handles to the step returns
         from all remote disciples
      '''
      arg, rets = args, []
      for discIdx, disciple in enumerate(self.disciples):
         step = Ascend.localize(disciple.step, self.remote)

         arg = []
         for shardIdx, e in enumerate(args):
            if shard is None:
               arg = args
            elif shard[shardIdx]:
               arg.append(e[discIdx])
            else:
               arg.append(e)
         arg = tuple(arg)
                  
         rets.append(step(*arg))
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

   def step(self, *args, shrd=False):
      '''Synchronous wrapper around the step function
      function of all remote disciples

      Args:
         *args: broadcast to all disciples
 
      Returns:
         A list of returns from all disciples
      '''
      rets = self.distrib(*args)
      return self.sync(rets)

   def discipleLogs(self):
      logs = []
      for e in self.disciples:
         try:
            logs.append(e.logs.remote())
         except:
            logs.append(e.logs())

      try:
         logs = ray.get(logs)
      except:
         pass

      logs = Log.summary(logs)
      return logs

 
   def localize(f, remote):
      '''Converts to the correct local/remote function version

      Args:
         f: Function to localize
         remote: Whether f is remote
 
      Returns:
         A localized function (f or f.remote)
      '''
      return f if not remote else f.remote

   def isRemote(obj):
      '''Check if an object is remote'''
      return hasattr(obj, 'remote') or hasattr(obj, '__ray_checkpoint__')


