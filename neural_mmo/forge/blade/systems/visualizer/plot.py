from pdb import set_trace as T
import numpy as np

import time
import os
 
from collections import defaultdict

import ray

from neural_mmo.forge.blade.systems import visualizer
from neural_mmo.forge.blade.lib.enums import Neon, Solid

from tornado.ioloop import IOLoop
import bokeh, jinja2

#Adapted from https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
def rolling_window(a, window, pad=True):
   a = np.array(a)
   if pad:
      a = np.pad(a, pad_width=(window-1, 0), mode='edge')
   shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
   strides = a.strides + (a.strides[-1],)
   return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def polar(r, frac): 
   angle = 2 * 3.1415927 * frac
   return r*np.sin(angle), r*np.cos(angle)
 
def flat(vals):
   flat = []
   for key, val in sorted(vals.items()):
      flat += val
   return flat

def disableAxis(axis):
   axis.major_label_text_font_size = '0pt'
   axis.major_tick_line_color      = None
   axis.minor_tick_line_color      = None
   axis.axis_line_color            = None

def blank(width, height, blob={}, ylabel=None, colors=[]):
   fig = bokeh.plotting.figure(
         plot_width=width,
         plot_height=height,
         toolbar_location=None)

   fig.grid[0].visible = False
   fig.grid[1].visible = False

   line = fig.line([0, 0], [0, 0])
   line.visible = False

   disableAxis(fig.xaxis)
   disableAxis(fig.yaxis)

   legend = {}
   for idx, (color, (key, source)) in enumerate(zip(colors, blob.items())):
      if key is None:
         key = ylabel
      glyph = fig.circle([0], [0], name=key, size=0, color=color)
      legend[key] = [glyph]

   return fig, legend

class Plot:
   def __init__(self, config, fig, key, n):
      self.config = config
      self.fig    = fig
      self.key    = key
      self.n      = n

   def render(self, blob, colors):
      legend = []
      for idx, (color, (key, source)) in enumerate(zip(colors, blob.items())):
         if key is None:
            key = self.key

         legend.append(self.glyphs(source, key, color, idx))

      return legend

   def glyphs(self, source, key, color, idx):
      source = self.preprocess(source, key, color, idx)
      source = bokeh.plotting.ColumnDataSource(source)
      return self.plot(source, key, color, idx)

class MarginPlot(Plot):
   def __init__(self, config, fig, key, n, margin=0.05):
      super().__init__(config, fig, key, n)
      self.fig.y_range.range_padding = margin
      self.fig.x_range.range_padding = margin
 
class Training(Plot):
   def preprocess(self, data, key, color, idx, w=20):
      mean                   = np.array(data['Mean'])
      std                    = np.array(data['Std'])

      preprocessed           = {}
      preprocessed['x']      = data['x']
      preprocessed['smooth'] = mean
      preprocessed['lower']  = mean - std
      preprocessed['upper']  = mean + std

      return preprocessed

   def plot(self, source, key, color, idx):
      self.fig.xaxis.axis_label = 'Environment Episodes'
      self.fig.yaxis.axis_label = 'Normalized Value'

      line = self.fig.line(
         source=source,
         name=key,
         x='x',
         y='smooth',
         color=color)

      return [line]


class Line(Plot):
   def preprocess(self, data, key, color, idx, w=20):
      sliding = rolling_window(flat(data), w)
      smooth  = np.mean(sliding, axis=1)
      std     = np.std(sliding, axis=1)

      preprocessed           = {}
      preprocessed['x']      = np.arange(len(smooth))
      preprocessed['smooth'] = smooth
      preprocessed['lower']  = smooth - std
      preprocessed['upper']  = smooth + std

      return preprocessed

   def plot(self, source, key, color, idx):
      self.fig.xaxis.axis_label = 'Index in Logs'
      self.fig.yaxis.axis_label = 'Value'

      band = self.fig.varea(
         source=source,
         name=key,
         x='x',
         y1='lower',
         y2='upper',
         fill_color=color)

      lower = self.fig.line(
         source=source,
         name=key,
         x='x',
         y='lower',
         line_color=color,
         line_width=0.5,
         line_alpha=0.7,
         color=color)

      upper = self.fig.line(
         source=source,
         name=key,
         x='x',
         y='upper',
         line_color=color,
         line_width=0.5,
         line_alpha=0.7,
         color=color)

      line = self.fig.line(
         source=source,
         name=key,
         x='x',
         y='smooth',
         color=color)

      return [band, lower, upper, line]

class Scatter(MarginPlot):
   def preprocess(self, data, key, color, idx, w=20):
      preprocessed = defaultdict(list)

      for t, vList in data.items():
         for val in vList:
            preprocessed['x'].append(t)
            preprocessed['y'].append(val)

      return preprocessed

   def plot(self, source, key, color, idx):
      self.fig.xaxis.axis_label = 'Game Tick'
      self.fig.yaxis.axis_label = 'Value'

      circle = self.fig.circle(
         source=source,
         name=key,
         x='x',
         y='y',
         color=color)

      return [circle]


class Histogram(Plot):
   def preprocess(self, data, key, color, idx, nBins=25):
      hist, edges = np.histogram(flat(data), bins=nBins)
         
      preprocessed          = {}
      preprocessed['top']   = hist
      preprocessed['left']  = edges[:-1]
      preprocessed['right'] = edges[1:]
 
      return preprocessed

   def plot(self, source, key, color, idx):
      self.fig.xaxis.axis_label = 'Value'
      self.fig.yaxis.axis_label = 'Count'

      quad = self.fig.quad(
         source=source,
         name=key,
         top='top',
         bottom=0,
         left='left',
         right='right',
         fill_color=color,
         line_color=color,
         )

      return [quad]

class Gantt(Plot):
   def preprocess(self, data, key, color, idx):
      preprocessed = []
      
      for tick, vList in sorted(data.items()):
         for val in vList:
            preprocessed.append((tick-val, tick))

      preprocessed = sorted(preprocessed)
      bottom, top  = list(zip(*preprocessed))
      bottom, top  = np.array(bottom), np.array(top)

      preprocessed           = {}
      preprocessed['top']    = top
      preprocessed['bottom'] = bottom
      preprocessed['left']   = bottom - 1
      preprocessed['right']  = bottom

      return preprocessed

   def plot(self, source, key, color, idx):
      self.fig.xaxis.axis_label = 'Game Tick'
      self.fig.yaxis.axis_label = 'Range'

      quad = self.fig.quad(
         source=source,
         name=key,
         top='top',
         bottom='bottom',
         left='left',
         right='right',
         fill_color=color,
         line_color=None,
         )

      return [quad]

class Stats(Plot):
   def preprocess(self, data, key, color, idx):
      data  = flat(data)
      mmean = np.mean(data)
      mmin  = np.min(data)
      mmax  = np.max(data)

      preprocessed           = {}
      preprocessed['top']    = [mmax]
      preprocessed['mean']   = [mmean]
      preprocessed['bottom'] = [mmin]
      preprocessed['left']   = [idx]
      preprocessed['right']  = [idx+1]

      return preprocessed

   def plot(self, source, key, color, idx):
      self.fig.xaxis.axis_label = 'Value'
      self.fig.yaxis.axis_label = 'Stat'
      disableAxis(self.fig.xaxis)

      quad = self.fig.quad(
         source=source,
         name=key,
         top='top',
         bottom='bottom',
         left='left',
         right='right',
         fill_color=color,
         line_color=color,
         )

      quad = self.fig.quad(
         source=source,
         name=key,
         top='mean',
         bottom='mean',
         left='left',
         right='right',
         fill_color=color,
         line_color=color,
         line_width=3.0,
         fill_alpha=1.0
         )


      return [quad]

class Radar(MarginPlot):
   def preprocess(self, data, key, color, idx):
      data  = flat(data)
      mmean = np.mean(data)
      mmin  = np.min(data)
      mmax  = np.max(data)

      preprocessed           = {}
      preprocessed['top']    = [mmax]
      preprocessed['mean']   = [mmean]
      preprocessed['bottom'] = [mmin]
      preprocessed['left']   = [idx]
      preprocessed['right']  = [idx+1]

      return preprocessed


   def render(self, blob, colors):
       self.fig.xaxis.axis_label = 'Radar Axis'
       self.fig.yaxis.axis_label = 'Radar Axis'
       disableAxis(self.fig.xaxis)
       disableAxis(self.fig.yaxis)

       legend, data = {}, defaultdict(list)
       for idx, (color, (subkey, source)) in enumerate(zip(colors, blob.items())):
          for key, val in self.preprocess(source, None, color, idx).items():
             data[key] += val

       n = len(blob)

       radians = 2 * 3.14159265

       #data['top']    = np.array(data['top'])*0 + 10
       #data['mean']   = np.array(data['mean'])*0 + 5
       #data['bottom'] = np.array(data['bottom'])*0 + 2

       for idx, (color, (subkey, source)) in enumerate(zip(colors, blob.items())):
          if subkey is None:
             subkey = key


          lIdx   = (idx - 1) % n
          mIdx   = idx % n
          rIdx   = (idx + 1) % n

          xlb, ylb = polar(data['bottom'][lIdx],  lIdx/n)
          xmb, ymb = polar(data['bottom'][mIdx],  mIdx/n)
          xrb, yrb = polar(data['bottom'][rIdx], rIdx/n)

          xlm, ylm = polar(data['mean'][lIdx],  lIdx/n)
          xmm, ymm = polar(data['mean'][mIdx],  mIdx/n)
          xrm, yrm = polar(data['mean'][rIdx],  rIdx/n)

          xlt, ylt = polar(data['top'][lIdx],  lIdx/n)
          xmt, ymt = polar(data['top'][mIdx],  mIdx/n)
          xrt, yrt = polar(data['top'][rIdx], rIdx/n)


          xlb, ylb = (xlb+xmb)/2, (ylb+ymb)/2
          xrb, yrb = (xmb+xrb)/2, (ymb+yrb)/2

          xlm, ylm = (xlm+xmm)/2, (ylm+ymm)/2
          xrm, yrm = (xmm+xrm)/2, (ymm+yrm)/2

          xlt, ylt = (xlt+xmt)/2, (ylt+ymt)/2
          xrt, yrt = (xmt+xrt)/2, (ymt+yrt)/2


          x = [xrb, xmb, xlb, xlt, xmt, xrt]
          y = [yrb, ymb, ylb, ylt, ymt, yrt]
          source = {'x': x, 'y': y}
          source = bokeh.plotting.ColumnDataSource(source)
          
          quad = self.fig.patch(
            source=source,
            name=key,
            x='x',
            y='y',
            fill_color=color,
            line_color=color,
            line_width=0.0,
            )

          x = [xlm, xmm, xrm]
          y = [ylm, ymm, yrm]
          source = {'x': x, 'y': y}
          source = bokeh.plotting.ColumnDataSource(source)
 
          mean = self.fig.line(
            source=source,
            name=key,
            x='x',
            y='y',
            line_width=2.5,
            color=color,)

          x = [xlb, xmb, xrb]
          y = [ylb, ymb, yrb]
          source = {'x': x, 'y': y}
          source = bokeh.plotting.ColumnDataSource(source)
 
          bottom = self.fig.line(
            source=source,
            name=key,
            x='x',
            y='y',
            line_width=2.5,
            color=color,)

          x = [xlt, xmt, xrt]
          y = [ylt, ymt, yrt]
          source = {'x': x, 'y': y}
          source = bokeh.plotting.ColumnDataSource(source)
 
          top = self.fig.line(
            source=source,
            name=key,
            x='x',
            y='y',
            line_width=2.5,
            color=color,)


          legend[subkey] = [quad, mean, bottom, top]

       return legend

class StackedArea(Plot):
   def preprocess(self, data, key, color, idx, w=20):
      sliding = rolling_window(flat(data), w)
      smooth  = np.mean(sliding, axis=1)
      std     = np.std(sliding, axis=1)

      preprocessed           = {}
      preprocessed['x']      = np.arange(len(smooth))
      preprocessed['smooth'] = smooth
      preprocessed['lower']  = smooth - std
      preprocessed['upper']  = smooth + std

      return preprocessed

   def render(self, blob, colors):
       self.fig.xaxis.axis_label = 'Index in Logs'
       self.fig.yaxis.axis_label = 'Value'

       legend, bottom = {}, None
       for idx, (color, (subkey, source)) in enumerate(zip(colors, blob.items())):
          if subkey is None:
             subkey = self.key

          data = self.preprocess(source, None, color, idx)

          x = np.array(data['x'])
          y = np.array(data['smooth'])

          if bottom is None:
             bottom = x * 0.0
        
          data   = {'x': x, 'y1': bottom, 'y2': bottom + y}
          source = bokeh.plotting.ColumnDataSource(data)
          bottom = bottom + y
          
          band = self.fig.varea(
             source=source,
             name=subkey,
             x='x',
             y1='y1',
             y2='y2',
             fill_color=color)

          lower = self.fig.line(
             source=source,
             name=subkey,
             x='x',
             y='y1',
             line_color=color,
             line_width=0.5,
             color=color)

          upper = self.fig.line(
             source=source,
             name=subkey,
             x='x',
             y='y2',
             line_color=color,
             line_width=0.5,
             color=color)

          legend[subkey] = [band, lower, upper]

       return legend
