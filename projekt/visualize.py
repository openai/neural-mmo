from pdb import set_trace as T
import numpy as np

import time
import os
import sys
 
from collections import defaultdict
import json

import ray

from forge.blade.systems import visualizer
from forge.blade.systems.visualizer import plot, theme
from forge.blade.lib.enums import Neon, Solid
from forge.blade.lib.log import Quill

from tornado.ioloop import IOLoop
import bokeh, jinja2

def load(f):
   return np.load(f, allow_pickle=True).tolist()

#Adapted from https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
def rolling_window(a, window, pad=True):
   a = np.array(a)
   if pad:
      a = np.pad(a, pad_width=(window-1, 0), mode='edge')
   shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
   strides = a.strides + (a.strides[-1],)
   return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def visualize(config):
   fPath = os.path.join(config.LOG_DIR, config.LOG_FILE)
   data  = load(fPath)

   BokehServer(config, data).start()

class BokehServer:
   def __init__(self, config, data):
      self.config = config
      self.data   = data

   def start(self):
      server = bokeh.server.server.Server(
         {'/': self.visualize},
         port=self.config.PORT,
         num_procs=1)

      server.start()
      server.io_loop.add_callback(server.show, '/')
      server.io_loop.start()
  
   def visualize(self, doc):
      #Load Theme
      config = self.config
      if config.THEME_NAME == 'web':
         index  = config.THEME_WEB_INDEX
         preset = theme.Web
         colors = Neon.color12()
      elif config.THEME_NAME == 'publication':
         index  = config.THEME_PUBLICATION_INDEX
         preset = theme.Publication
         colors = Solid.color10()
      else: 
         sys.exit('THEME_NAME invalid: web or publication')

      self.colors = [bokeh.colors.RGB(*(c.rgb)) for c in colors]
      fPath       = os.path.join(config.THEME_DIR, config.THEME_FILE) 
      with open(fPath, 'w') as f:
         json.dump({'attrs': preset.dict()}, f)
      doc.theme = bokeh.themes.Theme(fPath)

      #Load Index
      with open(os.path.join(config.THEME_DIR, index)) as f:
         doc.template = jinja2.environment.Template(f.read())

      #Draw Plots
      bokeh.plotting.output_file(fPath)
      quill, layout = self.data[0], []

      import itertools
      lifetimes = list(itertools.chain(*quill[('Lifetime', (0,1,2,3))][None].values()))
      mean, std = np.mean(lifetimes), np.std(lifetimes)
      print("Lifetime:: Mean: {}, Std: {}".format(mean, std))

      for (key, plots), blob in quill.items():
         row, n = [], len(plots)
         for idx, plot in enumerate(plots):
            fig, legend = self.plot(key, blob, plot, idx, n)
            row.append(fig)

         layout.append(row)
      doc.add_root(bokeh.layouts.layout(layout))

   def plot(self, title, blob, plot, idx, n):
      config   = self.config
      TOOLTIPS = [
         ("Track", "$name"),
         ("(x,y)", "($x, $y)"),
      ]

      offset = 228 #Magic number: width of title + legend
      width  = (config.PLOT_WIDTH - offset) // n
      if (config.PLOT_INTERACTIVE or idx != 0):
         offset = 0

      #Draw figure
      fig = bokeh.plotting.figure(
         plot_width= width+offset,
         plot_height=config.PLOT_HEIGHT,
         tools='hover, pan, box_zoom, wheel_zoom, reset, save',
         active_drag='pan',
         active_scroll='wheel_zoom',
         active_inspect='hover',
         tooltips=TOOLTIPS,
         title_location='above',
         y_axis_location='right',
         )

      fig.y_range.range_padding = 0.0
      fig.x_range.range_padding = 0.0

      fig.axis.axis_label_text_font_style = 'bold'
      fig.toolbar.logo     = None
      if not config.PLOT_TOOLS:
         fig.toolbar_location = None

      #Add glyphs
      Plot, n = Quill.plot(plot), len(blob)
      legend  = Plot(config, fig, title, n).render(blob, self.colors)
 
      #Add legend and title
      items = []
      for key, val in legend.items():
         items.append((key, val))

      legend   = bokeh.models.Legend(items=items, location='center')
      titleObj = bokeh.models.Title(text=title, align='center', vertical_align='middle')

      #Theme-specific formatting
      if config.PLOT_INTERACTIVE:
         titleObj.text           = title + ' vs. ' + fig.title.text
         titleObj.text_font_size = '12pt'
         fig.title.text          = None

         legend.orientation          ='horizontal'
         legend.label_text_font_size = '8pt'

         fig.add_layout(titleObj, 'left')
         fig.add_layout(legend, 'above')
      else:
         fig.min_border_right = 75
         legend.label_width   = 120
         if idx == 0:
            fig.add_layout(titleObj, 'left')
            fig.add_layout(legend, 'left')

      return fig, legend
