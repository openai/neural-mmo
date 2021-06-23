from pdb import set_trace as T

import numpy as np
import sys
import os
 
from neural_mmo.forge.blade.systems.visualizer import plot, theme
from neural_mmo.forge.blade.lib.log import Quill
from neural_mmo.forge.trinity import formatting

import bokeh, jinja2

class BokehServer:
   '''Interactive dashboard for visualizing evaluation logs

   This dashboard is created automatically from the data and plot types
   collected in Env.log. It supports both web and publication themes.'''
   def __init__(self, config):
      self.config = config

      server = bokeh.server.server.Server(
         {'/': self.visualize},
         port=self.config.VIS_PORT,
         num_procs=1)

      server.start()
      server.io_loop.add_callback(server.show, '/')
      server.io_loop.start()

   def load_theme(self, doc):
      '''Load dashboard theme'''
      config = self.config
      if config.VIS_THEME == 'web':
         preset = theme.Web(config)
      elif config.VIS_THEME == 'publication':
         preset = theme.Publication(config)
      else: 
         sys.exit('THEME_NAME invalid: web or publication')

      self.colors = [bokeh.colors.RGB(*(c.rgb)) for c in preset.colors]

      doc.theme = bokeh.themes.Theme(json={'attrs': preset.dict()})
      with open(preset.index) as f:
         doc.template = jinja2.environment.Template(f.read())

   def load_data(self):
      train_data, eval_data = {}, {}
      config = self.config

      train_path = config.PATH_TRAINING_DATA.format(config.MODEL)
      if os.path.exists(train_path):
         train_data = np.load(train_path, allow_pickle=True).tolist()['logs']
      else:
         print('VISUALIZE: No training logs for specified config')

      eval_path = config.PATH_EVALUATION
      if os.path.exists(eval_path):
         eval_data = np.load(eval_path, allow_pickle=True).tolist()
      else:
         print('VISUALIZE: No evaluation logs for specified config')

      return train_data, eval_data

   def resample(self, data, n):
      config  = self.config
      n      -= 1

      for track, stats in data.items():
         for key, vals in stats.items():
            llen   = len(vals) - 1
            window =  llen // n - 1
            sz     = window * n
            ary    = np.array(vals[1:sz+1]).reshape(n, -1)

            first = vals[0]
            mid   = ary[:-1].mean(1).tolist()
            last  = np.mean(ary[-1].tolist() + vals[sz+1:])
         
            ary        = np.array([first] + mid + [last])
            mmin, mmax = np.min(ary), np.max(ary)
            ary        = (ary - mmin) / (mmax - mmin)

            data[track][key] = ary.tolist()
         data[track]['x'] = [1] + (1 + window*np.arange(1, n)).tolist() + [len(vals)]
 
   def visualize(self, doc):
      '''Build dashboard'''
      config = self.config
      self.load_theme(doc)

      #Load data
      train_data, eval_data = self.load_data()
      if len(train_data) > 0:
         self.resample(train_data, config.TRAIN_DATA_RESAMPLE)
         train_data = {('Training', (-1,)): train_data}
      if len(eval_data) > 0:
         for line in formatting.table_stats(eval_data['Stats']):
            print(line)
         eval_data  = eval_data['Log'][0]

      layout = []
      for (key, plots), blob in {**train_data, **eval_data}.items():
         row, n = [], len(plots)

         #Left title
         titleObj = bokeh.models.Title(text=key, align='center', vertical_align='middle')
         fig, _   = plot.blank(60, config.VIS_HEIGHT)
         fig.add_layout(titleObj, 'left')
         row.append(fig)

         #Monochrome or full palette
         colors = self.colors
         if config.VIS_THEME == 'publication' and len(blob)==1:
            colors = [bokeh.colors.RGB(0, 0, 0)]
 
         #Draw plot row
         for idx, p in enumerate(plots):
            fig, legend = self.plot(key, blob, p, colors, idx, n)
            row.append(fig)
         
         #Right legend
         fig, legend = plot.blank(
               config.VIS_LEGEND_WIDTH + config.VIS_LEGEND_OFFSET,
               config.VIS_HEIGHT, blob, key, colors)
         fig.outline_line_width = 0
         fig.min_border_right   =0
         items  = list(legend.items())
         legend = bokeh.models.Legend(items=items, location='center')
         legend.label_width = config.VIS_LEGEND_WIDTH
         fig.add_layout(legend, 'right')
         row.append(fig)

         layout.append(row)

      
      doc.add_root(bokeh.layouts.layout(layout))
      return doc


   def plot(self, ylabel, blob, plot, colors, idx, n):
      config = self.config
      Plot   = Quill.plot(plot)
      title  = Plot.__name__

      #Correct dimensions
      width  = ((config.VIS_WIDTH - config.VIS_LEGEND_WIDTH
            - config.VIS_LEGEND_OFFSET - config.VIS_TITLE_OFFSET)  // n)
      height = config.VIS_HEIGHT
      aspect = height / width

      #Interactive plot element
      TOOLTIPS = [
         ("Track", "$name"),
         ("(x,y)", "($x, $y)"),
      ]

      #Make figure
      fig = bokeh.plotting.figure(
         plot_width=width,
         plot_height=height,
         x_range=bokeh.models.ranges.DataRange1d(range_padding=0.1*aspect),
         y_range=bokeh.models.ranges.DataRange1d(range_padding=0.1),
         tools='hover, pan, box_zoom, wheel_zoom, reset, save',
         active_drag='pan',
         active_scroll='wheel_zoom',
         active_inspect='hover',
         tooltips=TOOLTIPS,
         title_location='above',
         y_axis_label=ylabel,
         x_axis_location='below',
         min_border_right=config.VIS_BORDER_WIDTH,
         min_border_left=config.VIS_BORDER_WIDTH,
         min_border_top=config.VIS_BORDER_HEIGHT,
         min_border_bottom=config.VIS_BORDER_HEIGHT)

      #Extra options
      fig.axis.axis_label_text_font_style = 'bold'

      #Toolbars
      fig.toolbar.logo = None
      if not config.VIS_TOOLS:
         fig.toolbar_location = None
        
      #Draw glyphs
      legend = Plot(config, fig, ylabel, len(blob)).render(blob, colors)

      #Adjust for theme
      if config.VIS_THEME == 'web':
         fig.title.text = '{} vs. {}'.format(
               fig.yaxis.axis_label, fig.xaxis.axis_label)
         fig.yaxis.axis_label = None
         fig.xaxis.axis_label = None

      return fig, legend

