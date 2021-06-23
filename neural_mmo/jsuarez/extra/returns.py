import sys
import experiments
import numpy as np
import logs as loglib
from pdb import set_trace as T

if __name__ == '__main__':
   arg = None
   if len(sys.argv) > 1:
      arg = sys.argv[1]

   logDir  = 'resource/data/exps/'
   logName = 'logs.json'
   fName = 'frag.png'

   exp = loglib.load(logDir + 'combatscale/' + logName)
   rets = []
   for idx in range(32):
      ret = np.mean(exp['lifespan' + str(idx) + '_mean'][:-50])
      rets.append((ret, idx))
   rets = sorted(rets)
   print(rets)


