import numpy as np
import sys, json
from forge.blade.lib.enums import Neon, Color256
from forge.blade.lib.log import InkWell
from matplotlib import pyplot as plt
from pdb import set_trace as T
from itertools import groupby
from collections import defaultdict
import collections
import logs as loglib
import experiments
import pickle
import os.path as osp
import os

def plot(x, idxs, label, idx, path):
   colors = Neon.color12()
   loglib.dark()
   c = colors[idx % 12]
   loglib.plot(x, inds=idxs, label=str(idx), c=c.norm)
   loglib.godsword()
   loglib.save(path + label + '.png')
   plt.close()

def plots(x, label, idx, path, split):
   colors = Neon.color12()
   loglib.dark()
   for idx, item in enumerate(x.items()):
      annID, val = item
      c = colors[idx % 12]
      idxs, val = compress(val, split)
      loglib.plot(val, inds=idxs, label=str(annID), c=c.norm)
   loglib.godsword()
   loglib.save(path + label + '.png')
   plt.close()

def meanfilter(x, n=1):
   ret = []
   for idx in range(len(x) - n):
      val = np.mean(x[idx:(idx+n)])
      ret.append(val)
   return ret

def compress(x, split):
   rets, idxs = [], []
   if split == 'train':
      n = 1 + len(x) // 20
   else:
      n = 1 + len(x) // 20
   for idx in range(0, len(x) - n, n):
      rets.append(np.mean(x[idx:(idx+n)]))
      idxs.append(idx)
   return 10*np.array(idxs), rets

def popPlots(popLogs, path, split):
   idx = 0
   print(path)

   for key, val in popLogs.items():
      print(key)
      #val = meanfilter(val, 1+len(val)//100)
      plots(val, str(key), idx, path, split)
      idx += 1

def flip(popLogs):
   ret = defaultdict(dict)
   for annID, logs in popLogs.items():
      for key, log in logs.items():
         if annID not in ret[key]:
            ret[key][annID] = []
         if type(log) != list:
            ret[key][annID].append(log)
         else:
            ret[key][annID] += log
   return ret

def group(blobs, idmaps):
   rets = defaultdict(list)
   for blob in blobs:
      groupID = idmaps[blob.annID]
      rets[groupID].append(blob)
   return rets

def mergePops(blobs, idMap):
   #blobs = sorted(blobs, key=lambda x: x.annID)
   #blobs = dict(blobs)
   #idMap = {}
   #for idx, accumList in enumerate(accum):
   #   for e in accumList:
   #      idMap[e] = idx

   blobs = group(blobs, idMap)
   pops = defaultdict(list)
   for groupID, blobList in blobs.items():
      pops[groupID] += list(blobList)
   return pops
 
def individual(blobs, logDir, name, accum, split):
   savedir = logDir + name + '/' + split + '/'
   if not osp.exists(savedir):
      os.makedirs(savedir)

   blobs = mergePops(blobs, accum)
   popLogs = {}
   for annID, blobList in blobs.items():
      logs, blobList = {}, list(blobList)
      logs = {**logs, **InkWell.counts(blobList)}
      logs = {**logs, **InkWell.unique(blobList)}
      logs = {**logs, **InkWell.explore(blobList)}
      logs = {**logs, **InkWell.lifetime(blobList)}
      logs = {**logs, **InkWell.reward(blobList)}
      logs = {**logs, **InkWell.value(blobList)}
      popLogs[annID] = logs

   popLogs = flip(popLogs)
   popPlots(popLogs, savedir, split)

def makeAccum(config, form='single'):
   assert form in 'pops single split'.split()
   if form == 'pops':
      return dict((idx, idx) for idx in range(config.NPOP))
   elif form == 'single':
      return dict((idx, 0) for idx in range(config.NPOP))
   elif form == 'split':
      pop1 = dict((idx, 0) for idx in range(config.NPOP1))
      pop2 = dict((idx, 0) for idx in range(config.NPOP2))
      return {**pop1, **pop2}
   
if __name__ == '__main__':
   arg = None
   if len(sys.argv) > 1:
      arg = sys.argv[1]

   logDir  = 'resource/exps/'
   logName = '/model/logs.p'
   fName = 'frag.png'
   name = 'newfig'
   exps = []
   for name, config in experiments.exps.items():
      try:
         with open(logDir + name + logName, 'rb') as f:
            dat = []
            idx = 0
            while True:
               idx += 1
               try:
                  dat += pickle.load(f)
               except EOFError as e:
                  break
            print('Blob length: ', idx)
            split = 'test' if config.TEST else 'train'
            accum = makeAccum(config)
            individual(dat, logDir, name, accum, split)
            print('Log success: ', name)
      except Exception as err:
         print(str(err))



