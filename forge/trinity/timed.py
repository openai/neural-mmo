from pdb import set_trace as T
import time

from collections import defaultdict
import ray

def timed(func):
   def decorated(self, *args):
      t = time.time()
      ret = func(self, *args)
      t = time.time() - t
      self.log_time += t
      return ret

   return decorated

class Timed:
   def __init__(self):
      self.log_time = 0

   @property
   def time(self):
      time = self.log_time
      self.log_time = 0
      return time

   @timed
   def distrib(self, *args):
      return self.step(*args)

   @property
   def name(self):
      return self.__class__.__name__

   def logs(self):
      ret = Log(self.name, self.time)
      if hasattr(self, 'disciples'):
         for e in self.disciples:
            try:
               val = ray.get(e.logs.remote())
            except:
               val = e.logs()
            ret.disciples.append(val)
      return ret

class Log:
   def __init__(self, cls, time):
      self.cls = cls
      self.time = time
      self.disciples = []

class Summary:
   def __init__(self, log):
      self.total    = log['Total']
      self.pantheon = log['Pantheon']
      self.god      = log['God']
      self.sword    = log['Sword']
      self.realm    = log['VecEnvRealm']

   def __str__(self):
      total    = 'Total: {0:.2f}, '.format(self.total)
      pantheon = 'Pantheon: {0:.2f}, '.format(self.pantheon)
      god      = 'God: {0:.2f}, '.format(self.god)
      sword    = 'Sword: {0:.2f}, '.format(self.sword)
      realm    = 'Realm: {0:.2f}'.format(self.realm)
      return total + pantheon + god + sword + realm

class TimeLog:
   def flatten(log):
      ret = []
      for e in log.disciples:
         ret += TimeLog.flatten(e)
      return [log] + ret

   #Todo: log class for merging. Basic + detailed breakdown.
   def flat(timedList):
      ret = defaultdict(list)
      timedList = TimeLog.flatten(timedList)
      for e in timedList:
         ret[e.cls].append(e.time)
      return ret

   def merge(log):
      t = [0]
      for d in log.disciples:
         t.append(d.time)
      log.time -= max(t)

      for d in log.disciples:
         TimeLog.merge(d)
 
   def log(timedList, mode='basic'):
      ret = {}
      ret['Total'] = timedList.time
      TimeLog.merge(timedList)
      logs = TimeLog.flat(timedList)
      for k, v in logs.items():
         if mode == 'basic':
            ret[k] = max(v)
         elif mode == 'advanced':
            ret[k] = {'min': min(v), 'max': max(v), 'avg': sum(v)/len(v)}

      summary = Summary(ret)
      print(str(summary))
      return ret

   def basic(timedList):
      return TimeLog.log(timedList, 'basic')

   def default(timedList):
      return TimeLog.log(timedList, 'default')

   def advanced(timedList):
      return TimeLog.log(timedList, 'advanced')

     
 

