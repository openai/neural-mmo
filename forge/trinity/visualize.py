from pdb import set_trace as T

import numpy as np
import sys
 
from forge.blade.systems.visualizer import plot, theme
from forge.blade.lib.log import Quill
from forge.trinity import formatting

import bokeh, jinja2

class BokehServer:
   '''Interactive dashboard for visualizing evaluation logs

   This dashboard is created automatically from the data and plot types
   collected in Env.log. It supports both web and publication themes.'''
   def __init__(self, config):
      self.data   = np.load(config.PATH_EVAL_DATA, allow_pickle=True).tolist()
      self.config = config
      self.start()

   def start(self):
      '''Start Bokeh server'''
      server = bokeh.server.server.Server(
         {'/': self.visualize},
         port=self.config.VIS_PORT,
         num_procs=1)

      server.start()
      server.io_loop.add_callback(server.show, '/')
      server.io_loop.start()

   def load(self):
      '''Load dashboard theme'''
      config = self.config
      if config.VIS_THEME == 'web':
         preset = theme.Web(config)
      elif config.VIS_THEME == 'publication':
         preset = theme.Publication(config)
      else: 
         sys.exit('THEME_NAME invalid: web or publication')

      self.colors = [bokeh.colors.RGB(*(c.rgb)) for c in preset.colors]
      return preset
 
   def visualize(self, doc):
      '''Build dashboard'''
      config = self.config

      #Load Theme and Index
      preset    = self.load()
      doc.theme = bokeh.themes.Theme(json={'attrs': preset.dict()})
      with open(preset.index) as f:
         doc.template = jinja2.environment.Template(f.read())

      #Draw Plots
      data   = self.data
      logs   = data['Log'][0]
      stats  = data['Stats']
      
      #Print stats
      for line in formatting.box_stats(stats):
         print(line)

      layout = []
      for (key, plots), blob in logs.items():
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
