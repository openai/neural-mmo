from pdb import set_trace as T

import time
import ray

from functools import partial
from threading import Thread
from random import random

from bokeh.server.server import Server
from bokeh.models import ColumnDataSource
from bokeh.models import Band
from bokeh.plotting import curdoc, figure, show
from bokeh.themes import Theme

from tornado import gen
from tornado.ioloop import IOLoop

from neural_mmo.forge.blade.core.market.config import *

# to run a demo with dummy data:
# $ bokeh serve --show visualizer.py

# market plans
# click on objects in market to display stats about them
# market overview tab
# -> trade bandwidth, demand, gdp

PORT=5009

class MarketVisualizer:
    def __init__(self, keys, history_len: int = 512,
                 title: str = "NeuralMMO Market Data", x: str = "tick",
                 ylabel: str = "Dummy Values"):
        """Visualizes a stream of data with threaded refreshing
    
        Args:
            keys        : List of object names (str) to be displayed on market
            history_len : How far back to plot data
            title       : Title of graph
            x           : Name of x axis data
            ylabel      : Name of y axis on plot
        """
        self.colors = 'blue red green yellow orange purple'.split()
        self.data        = {}

        self.history_len = history_len
        self.title       = title
        self.keys        = keys

        self.ylabel      = ylabel
        self.x           = x

        # TODO figure out a better way to ensure unique colors
        assert len(self.keys) <= len(self.colors), 'Limited color pool'

        for i, key in enumerate(self.keys):
            self.data[key] = []
            self.data[key+'lower'] = []
            self.data[key+'upper'] = []
        self.data['tick'] = []


    def init(self, doc):
        #Source must only be modified through a stream
        self.source = ColumnDataSource(data=self.data)

        #Enable theming
        theme = Theme('forge/blade/core/market/theme.yaml')
        doc.theme = theme
        self.doc = doc

        fig = figure(
           plot_width=600,
           plot_height=400,
           tools='pan,xwheel_zoom,box_zoom,save,reset',
           title='Neural MMO: Market Data',
           x_axis_label=self.x,
           y_axis_label=self.ylabel)

        #Initialize plots
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

        #Set root
        self.doc.add_root(fig)
        self.fig = fig

@ray.remote
class BokehServer:
    def __init__(self, market, *args, **kwargs):
        """ Runs an asynchronous Bokeh data streaming server
      
        Args:
           market : The market to visualize
           args   : Additional arguments
           kwargs : Additional keyword arguments
        """
        self.visu   = MarketVisualizer(*args, **kwargs)
        self.market = market

        server = Server(
                {'/': self.init},
                io_loop=IOLoop.current(),
                port=PORT,
                num_procs=1)

        self.server = server
        self.thread = None
        self.tick   = 0
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
            #Wait for thread to initialize
            time.sleep(0.05)
            if self.thread is None:
               continue 

            #Get remote market data
            packet = ray.get(self.market.getData.remote())
            if packet is None:
               continue

            #Ingest market data
            for key, val in packet.items():
              if key[-3:] == 'std':
                 key = key[:-4]
                 dat = packet[key]
                 self.visu.data[key + 'lower'].append(dat - val)
                 self.visu.data[key + 'upper'].append(dat + val)
              else:
                 self.visu.data[key].append(val)

            #Stream to Bokeh client
            self.doc.add_next_tick_callback(partial(self.stream))
            self.tick += 1

    @gen.coroutine
    def stream(self):
        '''Stream current data buffer to Bokeh client'''
        self.visu.source.stream(self.visu.data, self.visu.history_len)

@ray.remote
class Middleman:
    def __init__(self):
        '''Remote data buffer for two processes to dump and recv data

        This is probably not safe'''
        self.data = None
        
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
        self.data = data

class Market:
    def __init__(self, items, middleman):
        '''Dummy market emulator

        Args: 
           items     : List of item keys
           middleman : A Middleman object'''
        self.middleman = middleman
        self.items     = items

        self.keys = items
        self.data = {}
        self.tick = 0

        self.data['tick'] = 0
        for i, key in enumerate(self.keys):
            self.data[key] = 0

    def update(self):
        '''Updates market data and propagates to Bokeh server

        Note: best to update all at once. Current version may cause bugs'''
        for key, val in self.data.items():
            self.data[key] = val + 0.2*(random() - 0.5)
            if key == 'tick':
                self.data[key] = self.tick

        self.tick += 1
        self.middleman.setData.remote(self.data)

# Example setup
if __name__ == '__main__':
   ray.init()
   ITEMS = ['Food', 'Water', 'Health', 'Melee', 'Range', 'Mage']

   middleman  = Middleman.remote()
   market     = Market(ITEMS, middleman)
   visualizer = BokehServer.remote(middleman, ITEMS)

   while True:
     time.sleep(0.1)
     market.update()

