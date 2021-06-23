from pdb import set_trace as T

import time
import os
import ray
import sys
import json
import argparse
import pickle
import numpy as np

from scipy.interpolate import interp1d
from collections import defaultdict

from functools import partial
from threading import Thread
from random import random

import bokeh
from bokeh.server.server import Server
from bokeh.models import ColumnDataSource
from bokeh.models import Band
from bokeh.models.widgets import RadioButtonGroup
from bokeh.layouts import layout

from bokeh.plotting import curdoc, figure, show
from bokeh.themes import Theme

from tornado import gen
from tornado.ioloop import IOLoop

from neural_mmo.forge.blade.lib.enums import Neon
from neural_mmo.forge.blade.systems.visualizer.config import *
from signal import signal, SIGTERM, SIGINT
from time import gmtime, strftime

def pickle_write(content, name, append=1):
    'function to open file, pickle dump, then close'
    f = open(name, "ab") if append else open(name, "wb")
    pickle.dump(content, f)
    f.close()

def pickle_read(name):
    'function to open file, pickle load, then close'
    f = open(name, "rb")
    ret = pickle.load(f)
    f.close()
    return ret

#Adapted from https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
def rolling_window(a, window, pad=True):
    a = np.array(a)
    if pad:
      a = np.pad(a, pad_width=(window-1, 0), mode='edge')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class Analytics:
    def __init__(self, config):
        """Visualizes a stream of data with threaded refreshing. To
        add items, initialize using 'keys' kwarg or add to packet in
        stream()
        Args:
            keys        : List of object names (str) to be displayed on market
            history_len : How far back to plot data
            scales      : Scales for visualization
            title       : Title of graph
            x           : Name of x axis data
            ylabel      : Name of y axis on plot
        """
        self.colors = []
        for color in [Neon.GREEN, Neon.CYAN, Neon.BLUE]:
           color = bokeh.colors.RGB(*(color.rgb))
           self.colors.append(color)
         
        # Market stores data in a dictionary of lists
        # MarketVisualizer stores the data for each view as a dictionary,
        # inside another dictionary with keys of each scale.

        self.history_len = config.HISTORY_LEN
        self.title       = config.TITLE

        self.ylabel      = config.YLABEL
        self.x           = config.XAXIS
        self.XAXIS       = config.XAXIS
        self.scales      = config.SCALES
        self.scale       = config.SCALES[0]
        self.title       = config.TITLE
        self.log         = config.LOG
        self.load        = config.LOAD_EXP
        self.filename    = config.NAME

        self.data        = defaultdict(list)
        self.dataSource  = {}
        self.keys        = 'lifetime'.split()
        for key in self.keys:
           self.dataSource[key] = [1, 2]
           self.dataSource[key+'_x'] = [1, 2]
           self.dataSource[key+'_lower'] = [1, 2]
           self.dataSource[key+'_upper'] = [1, 2]
           self.dataSource[key+'_smooth'] = [1, 2]

    def init(self, doc):
        # Source must only be modified through a stream
        self.source = ColumnDataSource(data=self.dataSource)

        # Enable theming
        theme = Theme(os.path.dirname(os.path.abspath(__file__)) + '/theme.yaml')
        doc.theme = theme
        self.doc = doc

        fig = figure(
           plot_width=960,
           plot_height=540,
           tools='xpan,xwheel_zoom, xbox_zoom, reset, save',
           title=self.title,
           x_axis_label=self.XAXIS,
           y_axis_label=self.ylabel)

        # Initialize plots
        for i, key in enumerate(self.keys):
            band = Band(
               source=self.source,
               base=key + '_x',
               lower=key+'_lower',
               upper=key+'_upper',
               level='annotation',
               line_color=self.colors[i],
               line_width=1.0,
               line_alpha=0.7,
               fill_color=self.colors[i],
               fill_alpha=0.3)
            fig.add_layout(band) 

            fig.line(
                source=self.source,
                x=key+'_x',
                y=key+'_smooth',
                color=self.colors[i],
                legend_label=key)

        def switch_scale(attr, old, new):
            """Callback for RadioButtonGroup to switch tick scale
            and refresh document
            Args:
                attr: variable to be changed, in this case 'active'
                old: old index of active button
                new: new index of active button
            """
            self.scale = self.scales[new]
            self.source.data = self.data[self.scale]

        self.timescales = RadioButtonGroup(labels=[str(scale) for scale in self.scales],
                                           active=self.scales.index(self.scale))
        self.timescales.on_change('active', switch_scale)

        self.structure = layout([[self.timescales], [fig]])
        self.doc.add_root(self.structure)

        self.fig = fig

    def update(self, packet):
      for key, val in packet.items():
         assert type(val) == list, 'packet value must be a list'
         self.data[key] += val

      #Sort by length for aesthetic rendering order
      self.keys = sorted(self.keys, key=lambda k: -len(self.data[k]))
        
    def resample(self, nOut=350, w=20):
      for key in list(self.data.keys()):
         #Padded sliding window
         val     = self.data[key]
         sliding = rolling_window(val, w)

         #Stats
         smooth  = np.mean(sliding, axis=1)
         std     = np.std(sliding, axis=1)

         #Add keys
         self.data[key+'_x']      = np.arange(len(val))
         self.data[key+'_smooth'] = smooth
         self.data[key+'_lower']  = smooth - std
         self.data[key+'_upper']  = smooth + std
 
      for key, val in self.data.items():
         #Interpolate
         nIn            = len(val)
         xIn            = np.arange(nIn)
         xOut           = np.linspace(0, nIn, nOut)
         f              = interp1d( 
                              x=xIn, y=val,
                              kind='nearest',
                              fill_value='extrapolate')

         self.data[key] = f(xOut)

    def stream(self):
        '''Wrapper function for source.stream to enable
        adding new items mid-stream. Overwrite graph
        with new figure if packet has different keys.
        Args:
            packet: dictionary of singleton lists'''

        # Stream items & append data if no new entries

        self.dataSource = dict(self.data.copy())
        if self.log:
           pickle_write(self.dataSource, self.filename)
        
        self.source.stream(self.dataSource, self.history_len)

        # Refreshes document to add new item
        self.doc.remove_root(self.structure)
        self.init(self.doc)


@ray.remote
class BokehServer:
    def __init__(self, middleman, config):
        """ Runs an asynchronous Bokeh data streaming server.
      
        Args:
           market : The market to visualize
           args   : Additional arguments
           kwargs : Additional keyword arguments
        """
        self.analytics = Analytics(config)
        self.middleman = middleman
        self.thread    = None

        server = Server(
                {'/': self.init},
                io_loop=IOLoop.current(),
                port=config.PORT,
                num_procs=1)

        server.start()
        self.server = server
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()

    def init(self, doc):
        '''Initialize document and threaded update loop
        Args:
           doc: A Bokeh document
        '''
        self.analytics.init(doc)
        self.doc = doc

        self.thread = Thread(target=self.update, args=[])
        self.thread.start()
        self.started = True

    def update(self):
        '''Blocking update call to be run in a separate thread
        Ingests packets from a remote market and streams to Bokeh client'''
        self.n = 0
        while True:
            # Wait for thread to initialize
            time.sleep(0.05)
            if self.thread is None:
               continue 

            # Get remote market data
            if ray.get(self.middleman.getShutdown.remote()):
                self.middleman.setData.remote(self.analytics.data)
                sys.exit(0)

            packet = ray.get(self.middleman.getData.remote())
            if packet is None:
               continue

            self.analytics.update(packet)
            self.analytics.resample()

            # Stream to Bokeh client
            self.doc.add_next_tick_callback(partial(self.stream))

    @gen.coroutine
    def stream(self):
        '''Stream current data buffer to Bokeh client'''
        # visu is visualizer object
        self.analytics.stream()

@ray.remote
class Middleman:
    def __init__(self):
        '''Remote data buffer for two processes to dump and recv data.
        Interacts with Market and BokehServer.
        This is probably not safe'''
        self.data = None
        self.shutdown = 0

    def getData(self):
        '''Get data from buffer
        Returns:
           data: From buffer
        '''
        data = self.data
        self.data = None
        return data

    def setData(self, data):
        '''Set buffer data
        Args:
           data: To set buffer
        '''
        self.data = data.copy()

    # Notify remotes of visualizer shutdown
    def setShutdown(self):
        self.shutdown = 1

    # Retreive shutdown status
    def getShutdown(self):
        return self.shutdown

#class Market:
#    def __init__(self, items, middleman):
#        '''Dummy market emulator
#        Args: 
#           items     : List of item keys
#           middleman : A Middleman object'''
#        self.middleman = middleman
#        self.items     = items
#
#        self.keys = items
#        # Data contains market history for lowest values
#        self.data = {}
#
#        for i, key in enumerate(self.keys):
#            self.data[key] = 0
#
#    def update(self):
#        '''Updates market data and propagates to Bokeh server
#        # update data and send to middle man
#        Note: best to update all at once. Current version may cause bugs'''
#        for key, val in self.data.items():
#           self.data[key] = val + 0.2*(random() - 0.5)
#
#
#        # update if not shutting down visualizer
#        if not ray.get(middleman.getShutdown.remote()):
#            self.middleman.setData.remote(self.data)

