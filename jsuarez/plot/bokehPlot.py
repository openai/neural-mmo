from pdb import set_trace as T
import numpy as np
import time

import ray

from forge.blade.systems import visualizer

class Config:
   LOG         = False                       #Whether to enable data logging
   LOAD_EXP    = False                       #Whether to load from file
   NAME        = 'log'                       #Name of file to load/log to
   HISTORY_LEN = 100                         #Length of graph history
   XAXIS       = 'Training Epoch'            #Label of xaxis data values
   YLABEL      = 'Agent Lifetime'            #Label of data values
   TITLE       = 'Neural MMO v1.4 Baselines' #Figure title
   SCALES      = [1, 10, 100, 1000]          #Plot time scale
   PORT        = 5006

def load(f):
   return np.load(f).tolist()

if __name__ == '__main__':
   ray.init()
   config     = Config
   middleman  = visualizer.Middleman.remote()
   vis        = visualizer.BokehServer.remote(middleman, config)

   data = {
         'Convolutional': load('data_simple.npy'),
         'Recurrent': load('data_recur.npy'),
         'Attentional': load('data_attn.npy')}
   middleman.setData.remote(data)
   vis.update.remote()
   while True:
      time.sleep(1)
