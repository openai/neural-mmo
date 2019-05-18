from pdb import set_trace as T
import time

from collections import defaultdict
import ray

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
      if hasattr(self, 'disciples'):
         for e in self.disciples:
            try:
               val = ray.get(e.logs.remote())
            except:
               val = e.logs()
            ret.disciples.append(val)
      return ret

class Log:
   def __init__(self, cls, runTime, waitTime):
      self.cls  = cls
      self.run  = runTime
      self.wait = waitTime
      self.disciples = []

class Summary:
   def __init__(self, log, total):
      self.log   = log
      self.total = total

   def __str__(self):
      ret = ''
      keys = 'Pantheon God Sword Realm'.split()
      for key in keys:
         ret += '{0:<17}'.format(key)
      ret = '        ' + ret + '\n'

      for stat, log in self.log.items():
         line = '{0:<5}:: '.format(stat)
         for key, val in log.items():
            if key == 'Trinity':
               continue

            percent = 100 * val / self.total

            percent = '{0:.2f}%'.format(percent)
            val     = '({0:.2f}s)'.format(val)

            percent = '{0:<7}'.format(percent)
            val     = '{0:<10}'.format(val)

            line += percent + val

         line = line.strip()
         ret += line + '\n'
      return ret

class TimeLog:
   def flatten(log):
      ret = []
      for e in log.disciples:
         ret += TimeLog.flatten(e)
      return [log] + ret

   #Todo: log class for merging. Basic + detailed breakdown.
   def flat(timedList):
      ret = defaultdict(lambda: defaultdict(list))
      timedList = TimeLog.flatten(timedList)
      for e in timedList:
         ret['run'][e.cls].append(e.run)
         ret['wait'][e.cls].append(e.wait)
         ret['idle'][e.cls].append(e.idle)
      return ret

   #This function goes down and then back up
   #First applies max over subnodes. Then fills
   #each subnode with the parent node value.
   #This is to counteract the effect of the 
   #downwards pass pulling data up a layer.
   def merge(log, total, cache=(-1, -1)):
      run, wait = [0], [0]
      for d in log.disciples:
         run.append(d.run)
         wait.append(d.wait)

      run  = max(run)
      wait = max(wait)

      cacheRun, cacheWait = cache

      #Run time of the current layer
      log.run  = cacheRun - cacheWait

      #Time spent waiting for next layer to sync
      log.wait = cacheWait #- wait

      #Time spend idle while previous layer launches jobs
      log.idle = total - log.run - log.wait

      for d in log.disciples:
         TimeLog.merge(d, total, (run, wait))


   def log(timedList, mode='basic'):
      ret = defaultdict(dict)
      total = timedList.run
      TimeLog.merge(timedList, total)
      logs = TimeLog.flat(timedList)
      for stat, log in logs.items():
         for k, v in log.items():
            if mode == 'basic':
               ret[stat][k] = max(v)
            elif mode == 'advanced':
               ret[stat][k] = {
                     'min': min(v), 
                     'max': max(v), 
                     'avg': sum(v)/len(v)}

      summary = Summary(ret, total)
      print(str(summary))
      return ret

   def basic(timedList):
      return TimeLog.log(timedList, 'basic')

   def default(timedList):
      return TimeLog.log(timedList, 'default')

   def advanced(timedList):
      return TimeLog.log(timedList, 'advanced')

     
 

