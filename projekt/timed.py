'''Logging for the demo model

This was rewritten very quickly for the Ascend update.
It is still rough and will be updated soon'''

from pdb import set_trace as T

from collections import defaultdict
import time

from forge.trinity.ascend import Log

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
