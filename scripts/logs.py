import numpy as np
import sys
import json
from pdb import set_trace as T
from collections import defaultdict
from matplotlib import pyplot as plt
from forge.blade.lib.enums import Neon

def plot(data, inds=None, label='data', c=Neon.RED.norm, lw=3):
   if inds is None:
      inds = np.arange(len(data))
   plt.plot(inds, data, c=c, linewidth=lw, label=label)

def dark():
   plt.style.use('dark_background')

def labels(xlabel='x', ylabel='y', title='title',
         axsz=24, titlesz=28):
   plt.xlabel(xlabel, fontsize=axsz)
   plt.ylabel(ylabel, fontsize=axsz)
   plt.title(title, fontsize=titlesz)

def axes(ac, tc):
   ax = plt.gca()
   ax.title.set_color(ac)
   ax.xaxis.label.set_color(ac)
   ax.yaxis.label.set_color(ac)
   for spine in ax.spines.values():
      spine.set_color(tc)

def limits(xlims=None, ylims=None):
   if xlims is not None:
      plt.xlim(*xlims)

   if ylims is not None:
      plt.ylim(*ylims)

def ticks(ts, tc):
   ax = plt.gca()
   ax.tick_params(axis='x', colors=tc)
   ax.tick_params(axis='y', colors=tc)
   for tick in ax.xaxis.get_major_ticks():
      tick.label1.set_fontsize(ts)
      tick.label1.set_fontweight('bold')
   for tick in ax.yaxis.get_major_ticks():
      tick.label1.set_fontsize(ts)
      tick.label1.set_fontweight('bold')

def legend(ts, tc):
   leg = plt.legend(loc='upper right')
   for text in leg.get_texts():
       plt.setp(text, color=tc)
       plt.setp(text, fontsize=ts)

def fig():
   fig = plt.gcf()
   fig.set_size_inches(12, 8, forward=True)
   #plt.tight_layout()

def show():
   fig.canvas.set_window_title('Projekt Godsword')

def save(fPath):
   fig = plt.gcf()
   fig.savefig(fPath, dpi=300)

def load(fDir):
   try:
      with open(fDir, 'r') as f:
         logs = json.load(f)

      logDict = defaultdict(list)
      for log in logs:
         for k, v in log.items():
            logDict[k].append(v)
      return logDict
   except Exception as e:
      print(e)
      return None

def godsword():
   labels('Steps', 'Value', 'Projekt: Godsword')
   axes(Neon.MINT.norm, Neon.CYAN.norm)
   ticks(18, Neon.CYAN.norm)
   legend(18, Neon.CYAN.norm)
   fig()

def plots(logs):
   colors = Neon.color12()
   logs = reversed([e for e in logs.items()])
   for idx, kv in enumerate(logs):
      k, v = kv
      color = colors[idx].norm
      plot(v, k, color)

def log():
   fDir = 'resource/logs/'
   fName = 'frag.png'
   logs = load(fDir + 'logs.json')
   dark()
   plots(logs)
   plt.ylim(0, 150)
   godsword()
   save(fDir+fName)
   plt.close()

