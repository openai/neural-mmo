from pdb import set_trace as T
import sys, torch
import os

expPath = 'resource/data/exps/'
fName   = '/bests.pth'

def loadExp(path):
   return torch.load(path)['param']

def saveExp(dat, path, expName):
   if not os.path.exists(path):
      os.makedirs(path)
   torch.save({'param':dat, 'epoch':0}, path + expName)

def runMatches():
   sz = [8, 32, 64, 128]
   n = len(sz)
   for prefix in ('nlaw', 'nchaos'):
      for i in range(n):
         for j in range(i+1, n):
            v1, v2 = str(sz[i]), str(sz[j])
            runMatch(prefix, v1, v2)

def runMatch(prefix, v1, v2):
   #exp1 = loadExp(expPath + prefix + v1 + fName)
   #exp2 = loadExp(expPath + prefix + v2 + fName)

   prefix = 'adam'
   exp1 = loadExp(expPath + 'adamentlaw128' + fName)
   exp2 = loadExp(expPath + 'adamentchaos128' + fName)

   params = torch.cat((exp1, exp2))
   path = expPath + 'tourney_' + prefix + v1 + '_' + v2
   saveExp(params, path, fName)

 
#runMatches()
runMatch(None, '128', '128')


