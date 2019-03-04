import numpy as np
import sys, json
from scipy.ndimage.filters import convolve
from forge.blade.lib.enums import Neon, Color256
from forge.blade.lib.log import Well
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

#Trajectory equated runs
def law128():
   rets = {
    '16': [(67.16070920182207 + 67.27818177485464) / 2, 65.53879245987099, 53.50681153837636, 47.796987075763205],
    '32': [74.32249298595012, (74.23310069290461 + 74.37120932322682) / 2, 62.39473940496259, 52.534221364625466],
    '64': [93.80093851827966, 97.00682644982263, (90.09223534011689 + 87.19457911719405) / 2, 75.07034083273426],
    '128': [109.72334511905568, 121.50424928230957, 121.95939904929756, (103.88169608454398 + 109.89533168831244) / 2],
   }
   return rets

def law64():
   rets = {
    '16': [70.34231691439271, 68.08311250680488, 59.693583536433955, 53.73726866287061], 
    '32': [81.34780761917122, 77.11407163999272, 69.42298166302263, 64.08609402748493], 
    '64': [119.08108040007559, 128.70866058606734, 107.03441943941378, 96.49304033434062], 
    '128': [228.30000553907186, 248.20682367308666, 241.56711251586333, 176.9001893357281]
    }
   return rets

def law32():
   rets = {
   '16': [85.87269967542795, 81.46462612460303, 76.59887029973338, 69.42979843534654], 
   '32': [101.75715811220914, 97.42858762722568, 93.53564789260295, 82.37687552017428], 
   '64': [175.1085580402948, 166.6336066014243, 158.17929884080706, 150.65147472068676], 
   '128': [467.5510996383706, 452.08437078645403, 453.8226233743415, 343.7846252190053]
   }
   return rets

def law16():
   rets = {
   '16': [106.08449913178288, 104.23812888187803, 89.77015854798023, 91.83535796395272], 
   '32': [118.97521974482278, 112.70542221817735, 115.6731408617378, 106.67977814200242], 
   '64': [205.66711549698138, 225.70099427243377, 208.46919256447077, 186.89705040539604], 
   '128': [735.7536877976454, 785.5794464698289, 781.9688869034464, 595.1878608252023]
   }
   return rets

#Max len runs
def adamlaw128():
   rets = {
   '16':  [],
   '32':  [],
   '64':  [],
   '128': [],
   }

def plots(x, idxs, sz):
   plt.style.use('bmh')
   colors = Neon.color12()
   idx = 0
   for label, vals in x.items():
      #c = colors[idx % 12]
      plt.plot(idxs, vals, linewidth=5, linestyle='--', marker='o', markersize=15, markerfacecolor='None', label=str(label), markeredgewidth=3)
      idx += 1
   plt.grid(linestyle='--', linewidth=1)
   plt.xticks(idxs)
   loglib.ticks(24, Neon.BLACK.norm)
   loglib.legend(24, Neon.BLACK.norm)
   loglib.labels(xlabel='Opponent Population Size at Train Time', ylabel='Lifetime In Tournament', title='Tournament at Population Size '+str(sz), axsz=32, titlesz=36)
   loglib.fig()
   loglib.save('tourney'+str(sz)+'.png')
   plt.close()

#Rows: 16, 32, 64, 128
rets = [law16(), law32(), law64(), law128()]
idxs = [16, 32, 64, 128]
for idx, ret in zip(idxs, rets):
   plots(ret, idxs, idx)

