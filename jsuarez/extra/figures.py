import numpy as np
import sys, json
from forge.blade.lib.enums import Neon, Color256
from matplotlib import pyplot as plt
from pdb import set_trace as T
import logs as loglib
import experiments
import os.path as osp
import os

def gen_plot(log, keys, savename, train=True):
   loglib.dark()

   if len(keys) > 12:
      colors = Color256.colors
   else:
      colors = Neon.color12()

   pops = []
   for i, key in enumerate(keys):
      c = colors[i]

      if not train:
         log[key] = np.cumsum(np.array(log[key])) / (1+np.arange(len(log[key])))

      if i == 0:
         loglib.plot(log[key], key, (1.0, 0, 0))
      else:
         loglib.plot(log[key], key, c.norm)
   loglib.godsword()
   loglib.save(savename)
   plt.close()

def individual(log, label, npop, logDir='resource/data/exps/', train=True):

   if train:
      split = 'train'
   else:
      split = 'test'

   savedir = osp.join(logDir, label, split)
   if not osp.exists(savedir):
      os.makedirs(savedir)

   if len(log['return']) > 0:
      loglib.dark()
      keys   = reversed('return lifespan value value_loss pg_loss entropy grad_mean grad_std grad_min grad_max'.split())
      colors = Neon.color12()
      fName = 'frag.png'
      for idx, key in enumerate(keys):
         if idx == 0:
            c = colors[idx]
            loglib.plot(log[key], key, (1.0, 0, 0))
         else:
            c = colors[idx]
            loglib.plot(log[key], key, c.norm)
      maxLife = np.max(log['return'])
      loglib.limits(ylims=[0, 50*(1+maxLife//50)])
      loglib.godsword()
      savepath = osp.join(logDir, label, split, fName)
      loglib.save(savepath)
      print(savepath)
      plt.close()

   # Construct population specific code
   pop_mean_keys = ['lifespan{}_mean'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_mean.png')
   gen_plot(log, pop_mean_keys, savefile, train=train)

   # Per population movement probability
   pop_move_keys = ['pop{}_move'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_move.png')
   gen_plot(log, pop_move_keys, savefile, train=train)

   # Attack probability plots
   pop_move_keys = ['pop{}_range'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_range.png')
   gen_plot(log, pop_move_keys, savefile, train=train)

   pop_move_keys = ['pop{}_melee'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_melee.png')
   gen_plot(log, pop_move_keys, savefile, train=train)

   pop_move_keys = ['pop{}_mage'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_mage.png')
   gen_plot(log, pop_move_keys, savefile, train=train)

   # Movement tile entropy
   pop_move_keys = ['pop{}_entropy'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_move_entropy.png')
   gen_plot(log, pop_move_keys, savefile, train=train)

   # Population attack probabilities when action is selected
   pop_move_keys = ['pop{}_melee_logit'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_melee_logit.png')
   gen_plot(log, pop_move_keys, savefile, train=train)

   pop_move_keys = ['pop{}_range_logit'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_range_logit.png')
   gen_plot(log, pop_move_keys, savefile, train=train)

   pop_move_keys = ['pop{}_mage_logit'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_mage_logit.png')
   gen_plot(log, pop_move_keys, savefile, train=train)

   # Sum up all the logits to check if they actually sum to zero
   for i in range(npop):
      logit_sum = np.array(log['pop{}_melee_logit'.format(i)]) + np.array(log['pop{}_range_logit'.format(i)]) + np.array(log['pop{}_mage_logit'.format(i)])
      log['pop{}_sum_logit'.format(i)] = logit_sum

   pop_move_keys = ['pop{}_sum_logit'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_sum_logit.png')
   gen_plot(log, pop_move_keys, savefile, train=train)

   # Tile exploration statistics
   pop_move_keys = ['pop{}_grass_tiles'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_grass_tiles.png')
   gen_plot(log, pop_move_keys, savefile, train=train)

   pop_move_keys = ['pop{}_forest_tiles'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_forest_tiles.png')
   gen_plot(log, pop_move_keys, savefile, train=train)

   pop_move_keys = ['pop{}_forest_tiles_depleted'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_forest_depleted.png')
   gen_plot(log, pop_move_keys, savefile, train=train)

   # pop_move_keys = ['pop{}_forest_tiles_other'.format(i) for i in range(npop)]
   # savefile = osp.join(logDir, label, 'pop_forest_tiles_other.png')
   # gen_plot(log, pop_move_keys, savefile, train=train)

   for i in range(npop):
      forest_tiles = np.array(log['pop{}_forest_tiles'.format(i)])
      other_tiles = np.array(log['pop{}_grass_tiles'.format(i)]) + np.array(log['pop{}_forest_tiles_depleted'.format(i)]) + forest_tiles
      forage_percent = forest_tiles / other_tiles
      log['pop{}_forage_success'.format(i)] = forage_percent

   pop_move_keys = ['pop{}_forage_success'.format(i) for i in range(npop)]
   savefile = osp.join(logDir, label, split, 'pop_forage_success.png')
   gen_plot(log, pop_move_keys, savefile, train=train)


def individuals(exps):
   for name, npop, log in exps:
      try:
         individual(log, name, npop)
         print('Log success: ', name)
      except Exception as e:
         print(e)
         print('Log failure: ', name)

def joints(exps):
   print('Joints...')
   keys   = reversed('return lifespan value value_loss pg_loss entropy grad_mean grad_std grad_min grad_max'.split())
   colors = Neon.color12()
   for key in keys:
      loglib.dark()
      maxVal = 0
      for idx, dat in enumerate(exps):
         name, _, log = dat
         loglib.plot(log[key], name, colors[idx].norm, lw=3)
         maxVal = max(maxVal, np.max(log[key]))
      loglib.limits(ylims=[0, 50*(1+maxVal//50)])
      loglib.godsword()
      loglib.save(logDir+'joint/'+key)
      plt.close()

def agents():
   exps = list(experiments.exps.keys())
   loglib.dark()
   colors = Neon.color12()
   maxVal = 0
   for idx, exp in enumerate(exps):
      name, log = exp
      c = colors[idx]
      loglib.plot(log['lifespan'], name, c.norm)
      maxVal = max(maxVal, np.max(log['lifespan']))
   loglib.limits(ylims=[0, 50*(1+maxVal//50)])
   loglib.godsword()
   loglib.save(logDir+'/agents.png')
   plt.close()

def populations():
   pass

def combat():
   pass

if __name__ == '__main__':
   arg = None
   if len(sys.argv) > 1:
      arg = sys.argv[1]

   logDir  = 'resource/data/exps/'
   logName = 'logs.json'
   fName = 'frag.png'

   #exps = [(name, config.NPOP, loglib.load(logDir+name+'/'+logName))
   #      for name, config in experiments.exps.items()]

   exps = []
   for name, config in experiments.exps.items():
      try:
         exp = loglib.load(logDir + name + '/' + logName)
         individual(exp, name, config.NPOP)
         exps.append(exp)
         print('Log success: ', name)
      except Exception as e:
         print(e)
         print('Log failure: ', name)

   if arg == 'individual':
      individuals(exps)
   elif arg == 'joint':
      joints(exps)
   else:
      individuals(exps)
      joints(exps)

   # agents()
