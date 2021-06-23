from pdb import set_trace as T

import numpy as np


SEP     = u'\u2595\u258f'
BLOCK   = u'\u2591'
TOP     = u'\u2581'
BOT     = u'\u2594'
LEFT    = u'\u258f'
RIGHT   = u'\u2595'

def table_stats(stats, titleLen=12, entryLen=12):
   titleFmt = '{:<' + str(titleLen) + '}'
   valFmt   = '{:' + str(entryLen) + '.2f}'
   keyFmt   = '{:<' + str(entryLen) + '}'

   keys  = [keyFmt.format(k) for k in 'Min Max Mean Std'.split()]
   title = titleFmt.format('Metric')
   lines = [[title] + keys]

   for key, stat in stats.items():
      l = [titleFmt.format(key)]
      for func in (np.min, np.max, np.mean, np.std):
         val = valFmt.format(func(stat))
         l.append(val)
      lines.append(l)
      
   llens = [titleLen] + 4*[entryLen]
   seps  = ['='*l for l in llens]
   lines = [seps, lines[0], seps, *(lines[1:]), seps]
   lines = [' '.join(e) for e in lines]
 
   return lines


def precomputed_stats(stats):
   '''Format a dict of precomputed stats'''
   lines = []
   for key, stat in stats.items():
      keys  = 'Min Max Mean Std'.split()
      vals  = [stat[k] for k in keys]
      lines.append(line(title=key, keys=keys, vals=vals))
   
   return lines

def stats(stats):
   '''Format a dict of stats'''
   lines = []
   for key, stat in stats.items():
      mmin, mmax = np.min(stat),  np.max(stat)
      mean, std  = np.mean(stat), np.std(stat)

      lines.append(line(
            title = key,
            keys  = 'Min Max Mean Std'.split(),
            vals  = [mmin, mmax, mean, std]))

   return lines

def times(stats):
   '''Format a dict of timed data'''
   lines = []
   for key, stat in stats.items():
      ssum, n   = np.sum(stat),  len(stat)
      mean, std = np.mean(stat), np.std(stat)

      lines.append(line(
            title = key,
            keys  = 'Total N Mean Std'.split(),
            vals  = [ssum, n, mean, std]))

   return lines

def line(title=None, keys=[], vals=[],
      titleFmt='{:<12}', keyFmt='{}', valFmt='{:8.1f}'):
   '''Format a line of stats with vertical separators'''

   assert len(keys) == len(vals), 'Unequal number of keys and values'

   fmt = []
   if title is not None:
      fmt.append(titleFmt.format(title))

   for key, val in zip(keys, vals):
      fmt.append((keyFmt + ': ' + valFmt).format(key, val))

   return SEP.join(fmt)

def box(block, indent=1):
   '''Indent lines and draw a box around them'''
   mmax   = max(len(line) for line in block) + 2
   fmt    = '{:<'+str(mmax+1)+'}'
   indent = ' ' * indent

   for idx, l in enumerate(block):
      block[idx] = indent + fmt.format(LEFT + l + RIGHT)

   block.insert(0, indent + TOP*mmax)
   block.append(indent + BOT*mmax)
   return block

def box_stats(vals, indent=1):
   '''Format stats and then draw a box around them'''
   return box(stats(vals))
   
