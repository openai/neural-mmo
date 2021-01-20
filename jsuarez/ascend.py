from pdb import set_trace as T
import ray, time

from collections import defaultdict
import functools

class Timed:
   '''Performance timing superclass.

   Depends on runtime and waittime decorators'''
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
   '''Performance logging superclass

   Provides timing summaries over remote disciples'''
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
   @functools.wraps(func)
   def decorated(self, *args):
      t = time.time()
      ret = func(self, *args)
      t = time.time() - t
      self.wait_time += t
      return ret

   return decorated

def runtime(func):
   @functools.wraps(func)
   def decorated(self, *args):
      t = time.time()
      ret = func(self, *args)
      t = time.time() - t
      self.run_time += t
      return ret

   return decorated

class Ascend(Timed):
   '''This module is the Ascend core and only documents the internal API.
   External documentation is available at :mod:`forge.trinity.api`'''
   def __init__(self, disciple, n, *args):
      super().__init__()
      self.remote    = Ascend.isRemote(disciple)
      disciple       = Ascend.localize(disciple, self.remote)
      self.disciples = [disciple(*args, idx) for idx in range(n)]

   def distribute(self, *args, shard=None):
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
   def synchronize(self, rets):
      if self.remote:
         return ray.get(rets)
      return rets

   def step(self, *args, shard=False):
      rets = self.distribute(*args)
      return self.synchronize(rets)

   def discipleLogs(self):
      logs = []
      for e in self.disciples:
         log = e.logs
         try:
            log = ray.get(log.remote())
         except:
            log = log()
         logs.append(log)

      logs = Log.summary(logs)
      return logs

 
   def localize(func, remote):
      return func if not remote else func.remote

   def isRemote(obj):
      return hasattr(obj, 'remote') or hasattr(obj, '__ray_checkpoint__')
