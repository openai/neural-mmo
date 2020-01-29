from pdb import set_trace as T

import time
import os
import ray
import sys
import json
import argparse
import pickle
import numpy as np


from functools import partial
from threading import Thread
from random import random

from bokeh.server.server import Server
from bokeh.models import ColumnDataSource
from bokeh.models import Band
from bokeh.models.widgets import RadioButtonGroup
from bokeh.layouts import layout

from bokeh.plotting import curdoc, figure, show
from bokeh.themes import Theme

from tornado import gen
from tornado.ioloop import IOLoop

from forge.blade.systems.visualizer.config import *
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

class MarketVisualizer:
    def __init__(self, config, *args):
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
        self.colors = 'blue red green yellow orange purple brown white'.split()
        self.data        = {}

        # Market stores data in a dictionary of lists
        # MarketVisualizer stores the data for each view as a dictionary,

        # inside another dictionary with keys of each scale.

        self.history_len = config.HISTORY_LEN
        self.title       = config.TITLE

        self.ylabel      = config.YLABEL
        self.x           = config.XAXIS
        self.XAXIS       = config.XAXIS
        self.keys        = [self.x]
        self.scales      = config.SCALES
        self.title       = config.TITLE
        self.log         = config.LOG
        self.load        = config.LOAD_EXP
        self.filename    = config.NAME

        # set all scales to be strings instead of ints to match file reading
        if type(self.scales[0]) != str:
            self.scales = [str(s) for s in self.scales]
            if self.log:
                pickle_write(self.scales, self.filename, append=0)
                pickle_write(self.x, self.filename)

        # initialize blank data if no data loaded from file
        if not self.load:
            for scale in self.scales:
                self.data[scale] = {}
            for scale in self.scales:
                for i, key in enumerate(self.keys):
                    self.data[scale][key] = []
                    self.data[scale][key+'lower'] = []
                    self.data[scale][key+'upper'] = []
                self.data[scale][self.x] = []
                # Isn't necessary, but makes code for packet handling nicer
                self.data[scale][self.x + 'upper'] = []
                self.data[scale][self.x + 'lower'] = []
        else:
            # TODO write loading from pickle here
            file = open(self.filename, 'rb')
            self.scales = pickle.load(file)
            self.x      = pickle.load(file)
            parse = {}
            for scale in self.scales:
               parse[scale] = {}
            
            # combine data from all packets into relevant spots for scales
            while True:
               try: 
                  packet = pickle.load(file)
                  # add to each relevant scale
                  for scale in self.scales:
                      # skip over irrelevant scales
                      if packet[self.x][0] % int(scale) != 0:
                          continue
                      for key in packet: 
                          if key not in parse[scale]:
                              parse[scale][key] = [0] * (packet[self.x][0] // int(scale))
                          parse[scale][key].append(packet[key][0])
               except EOFError as e:                                    
                  break 

            self.data = parse.copy()
            self.keys = list(self.data[self.scales[0]].keys())
            self.keys.remove(self.x)
            self.keys = [k for k in self.keys if k[-5:] != 'upper' and k[-5:] != 'lower']

            data = np.load('lifetime.npy', allow_pickle=True).tolist()
            self.keys = list(data.keys())
            self.keys.remove(self.x)
            self.keys = [k for k in self.keys if k[-5:] != 'upper' and k[-5:] != 'lower']
            self.data = data

        # Set default open scale to leftmost scale
        self.scale = self.scales[0]

    def init(self, doc):
        # Source must only be modified through a stream
        self.source = ColumnDataSource(data=self.data)
        self.colors[0] = 'cyan'
        #self.source = ColumnDataSource(data=self.data[self.scale])

        # Enable theming
        theme = Theme(os.path.dirname(os.path.abspath(__file__)) + '/theme.yaml')
        doc.theme = theme
        self.doc = doc

        fig = figure(
           plot_width=600,
           plot_height=400,
           tools='xpan,xwheel_zoom, xbox_zoom, reset, save',
           title=self.title,
           x_axis_label=self.XAXIS,
           y_axis_label=self.ylabel)

        # Initialize plots
        for i, key in enumerate(self.keys):
            fig.line(
                source=self.source,
                x=self.x,
                y=key,
                color=self.colors[i],
                line_width=LINE_WIDTH,
                legend_label=key)

            band = Band(
               source=self.source,
               base=self.x,
               lower=key+'lower',
               upper=key+'upper',
               level='underlay',
               line_color=self.colors[i],
               line_width=1,
               line_alpha=0.2,
               fill_color=self.colors[i],
               fill_alpha=0.2)
            fig.add_layout(band) 

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

    def stream(self, packet: dict):
        '''Wrapper function for source.stream to enable
        adding new items mid-stream. Overwrite graph
        with new figure if packet has different keys.
        Args:
            packet: dictionary of singleton lists'''

        # Stream items & append data if no new entries

        if self.log:
           pickle_write(packet, self.filename)

        if packet.keys() == self.data[self.scale].keys():

            for scale in self.scales:
                # Skip over irrelevant scales
                if packet[self.x][0] % int(scale) != 0:
                    continue
                for key, val in packet.items():
                    self.data[scale][key].append(val[0])
            # Once data is added, stream if tick is member
            # of current scale
            if packet[self.x][0] % int(self.scale) == 0:
                self.source.stream(packet, self.history_len)
            return

        # Add new data entry & refresh document if new entry
        for scale in self.scales:
            for key, val in packet.items():

                # Pads new data with 0 entry before adding
                if key not in self.data[scale].keys():
                    pad_len = len(self.data[scale][self.x])
                    self.data[scale][key] = ([0] * pad_len)

                    # Adds new entry value if relevant
                    if packet[self.x][0] % int(scale) == 0 and pad_len > 0:
                        self.data[scale][key][-1] = val[0]
                    # Adds new entry to keys if valid key
                    if key[-5:] not in ['lower', 'upper'] and key not in self.keys + [self.x]:
                        self.keys.append(key)

                # If not new entry, add to data
                elif packet[self.x][0] % int(scale) != 0:
                    self.data[scale][key].append(val[0])

        # Refreshes document to add new item
        self.doc.remove_root(self.structure)
        self.init(self.doc)


@ray.remote
class BokehServer:
    def __init__(self, market, *args, **kwargs):
        """ Runs an asynchronous Bokeh data streaming server.
      
        Args:
           market : The market to visualize
           args   : Additional arguments
           kwargs : Additional keyword arguments
        """
        self.visu   = MarketVisualizer(*args, **kwargs)
        self.market = market
        if not 'PORT' in args:
            PORT = 5006

        server = Server(
                {'/': self.init},
                io_loop=IOLoop.current(),
                port=PORT,
                num_procs=1)

        self.server = server
        self.thread = None
        self.tick   = 0
        self.packet = {}

        if 'scales' in kwargs and type(kwargs['scales']) is list:
            self.scales = kwargs['scales']
        else:
            self.scales = [1]

        server.start()

        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()

    def init(self, doc):
        '''Initialize document and threaded update loop
        Args:
           doc: A Bokeh document
        '''
        self.visu.init(doc)
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

            if ray.get(self.market.getShutdown.remote()):
                self.market.setData.remote(self.visu.data)
                sys.exit(0)

            packet = ray.get(self.market.getData.remote())
            if packet is None:
               continue
            self.packet = packet.copy() if type(packet) == dict else {}

            # Ingest market data, add upper and lower values for each scale
            # self.packet streams latest tick to visualizer
            for key, val in packet.items():
                # Add to packet
                self.packet[key] = [self.packet[key]]
                self.packet[key + 'lower'] = [val - 0.1]
                self.packet[key + 'upper'] = [val + 0.1]

            # Stream to Bokeh client
            self.doc.add_next_tick_callback(partial(self.stream))
            self.tick += 1

    @gen.coroutine
    def stream(self):
        '''Stream current data buffer to Bokeh client'''
        # visu is visualizer object
        self.visu.stream(self.packet)

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

class Market:
    def __init__(self, items, middleman):
        '''Dummy market emulator
        Args: 
           items     : List of item keys
           middleman : A Middleman object'''
        self.middleman = middleman
        self.items     = items

        self.keys = items
        # Data contains market history for lowest values
        self.data = {}
        self.tick = 0

        self.data['tick'] = 0
        for i, key in enumerate(self.keys):
            self.data[key] = 0

    def update(self):
        '''Updates market data and propagates to Bokeh server
        # update data and send to middle man
        Note: best to update all at once. Current version may cause bugs'''
        for key, val in self.data.items():
            if key == 'tick':
                self.data[key] = self.tick
            else:
                self.data[key] = val + 0.2*(random() - 0.5)

        self.tick += 1

        # dummy code to inject new items, it's as easy as just
        # adding new data to the market

        if self.tick % 100 == 0:
            self.data['new_item'] = 5
        if self.tick % 130 == 0:
            self.data['new_item_2'] = 5

        # update if not shutting down visualizer
        if not ray.get(middleman.getShutdown.remote()):
            self.middleman.setData.remote(self.data)
