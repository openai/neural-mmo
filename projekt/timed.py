'''Logging for the demo model

This was rewritten very quickly for the Ascend update.
It is still rough and will be updated soon'''

from pdb import set_trace as T
import time

from collections import defaultdict
import ray

from forge.trinity.ascend import Ascend, Log

def format(x):
   return ('{0:<' + str(n) + '}').format(x)

class Summary:
   '''Formatted logging prints'''
   def __init__(self, log):
      self.log   = Log.aggregate(log)

   def __str__(self):
      ret = ''
      keys = 'Pantheon God Sword Realm'.split()
      for key in keys:
         ret += '{0:<17}'.format(key)
      ret = '        ' + ret + '\n'

      self.log['run']['God'] -= self.log['run']['Realm']
      total = self.log['run']['Pantheon'] + self.log['wait']['Pantheon']

      for stat, log in self.log.items():
         line = '{0:<5}:: '.format(stat)
         for key in keys: 
            val = log[key]

            percent = 100 * val / total
            percent = '{0:.2f}%'.format(percent)
            val     = '({0:.2f}s)'.format(val)

            percent = '{0:<7}'.format(percent)
            val     = '{0:<10}'.format(val)

            line += percent + val

         line = line.strip()
         ret += line + '\n'
      return ret

class TimeLog:
   def log(logs):
      keys = 'Pantheon God Sword'.split()
      for key, val in logs.items():
         log[key]
   
   #Todo: log class for merging. Basic + detailed breakdown.
   #This log function is just a hack for the v1.2 demo.
   #Will work out a better long term solution soon
   '''
   def log(trinity):
      logs = defaultdict(list)
      trinityLogs = trinity.logs()
      for pantheon in trinity.disciples:
         logs['Pantheon'].append(pantheon.logs())
         for god in pantheon.disciples:
            isRemote = Ascend.isRemote(god)
            godLogs   = Ascend.localize(god.logs, isRemote)()
            envLogs   = Ascend.localize(god.getEnvLogs, isRemote)()
            swordLogs = Ascend.localize(god.getSwordLogs, isRemote)()

            if isRemote:
                godLogs   = ray.get(godLogs)
                envLogs   = ray.get(envLogs)
                swordLogs = ray.get(ray.get(swordLogs))

            logs['God'].append(godLogs)
            logs['Sword'] += swordLogs
            logs['Realm'].append(envLogs)

      rets  = defaultdict(dict)
      total = trinityLogs.run
      for key, logList in logs.items():
         log = Log(key, 0, 0)

         run  = max([e.run for e in logList])
         wait = max([e.wait for e in logList])

         rets['run'][key]  = run - wait
         rets['wait'][key] = wait

      summary = Summary(rets, trinityLogs.run)
      print(str(summary))
      return rets
   '''
