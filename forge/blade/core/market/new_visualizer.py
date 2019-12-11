from pdb import set_trace as T

from functools import partial
from threading import Thread
from random import random
import time
import ray

from copy import deepcopy

from bokeh.server.server import Server
from bokeh.models import ColumnDataSource
# from bokeh.models.widgets import CheckboxGroup
from bokeh.plotting import curdoc, figure, show

from tornado import gen
from tornado.ioloop import IOLoop

from config import *

# to run a demo with dummy data:
# $ bokeh serve --show visualizer.py

# market plans
# click on objects in market to display stats about them
# market overview tab
# -> trade bandwidth, demand, gdp

class MarketVisualizer:
    """
    Market Visualizer
    Visualizes a stream of data, automatically refreshes on update()
    """
    def __init__(self, keys, history_len: int = 256,
                 title: str = "NeuralMMO Market Data", x: str = "tick",
                 ylabel: str = "Dummy Values"):
        """
        Args:
            keys (list):       List of object names (str) to be displayed on
                               the market
            history_len (int): How far back to plot data, default is 10 ticks
            title (str):       Title of graph
            x (str):           Name of x value, both in data representation and
                               x axis
            ylabel (str):      Name of y axis on plot
            seed (int):        seed for random number generation
        """
 
        self.COLORS = 'blue red green yellow black purple'.split()

        self.history_len = history_len
        self.title       = title
        self.keys        = keys

        self.ylabel      = ylabel
        self.x           = x

        self.data        = {}
        self.colors      = {}

    def init(self, doc):
        # TODO figure out a better way to ensure each 
        # object has a unique color
        assert len(self.keys) <= len(self.COLORS), 'Limited color pool'

        for i, key in enumerate(self.keys):
            self.data[key] = []
        self.data['tick'] = []

        # this must only be modified from a Bokeh session callback
        self.source = ColumnDataSource(data=self.data)

        # This is important! Save curdoc() to make sure all threads
        # see the same document.
        #self.doc = curdoc()
        self.doc = doc

        fig = figure(
           plot_width=600,
           plot_height=400,
           tools='xpan,xwheel_zoom, xbox_zoom, reset',
           title='Neural MMO: Market Data',
           x_axis_label=self.x,
           y_axis_label=self.ylabel,
           **PLOT_OPTS)

        for i, key in enumerate(self.keys):
            self.colors[key] = self.COLORS[i]
            fig.line(x=self.x, y=key, source=self.source,
                color=self.colors[key], legend_label=key)

        #fig.xaxis.ticker = scales                                                     
        #fig.xaxis.major_label_overrides = {1:'', 2:''}                                
                                                                                       
        fig.yaxis.major_tick_line_color='black'                                       
        fig.xaxis.major_tick_line_color='black'                                       
                                                                                       
        fig.title.text_font_size = TEXT_LARGE                                         
        fig.title.text_color = MAIN_COLOR                                             
        fig.title.align = 'center'                                                    
                                                                                       
        #fig.grid[0].ticker  = scales                                                  
        fig.legend.location = 'bottom_left'
        fig.legend.label_text_color     = 'black'
        fig.legend.label_text_font_size = TEXT_SMALL
                                                   
        fig.xaxis.major_label_text_font_size  = TEXT_SMALL
        fig.xaxis.major_label_text_color = MAIN_COLOR    
        fig.xaxis.axis_label_text_font_size = TEXT_MEDIUM
        fig.xaxis.axis_label_text_color = MAIN_COLOR
                                                                                       
        fig.yaxis.major_label_text_font_size = TEXT_SMALL
        fig.yaxis.major_label_text_color = MAIN_COLOR
        fig.yaxis.axis_label_text_font_size = TEXT_MEDIUM
        fig.yaxis.axis_label_text_color = MAIN_COLOR

        # Hide lines that you click on in legend,
        # ease of use for visualization
        fig.legend.click_policy = "hide"
        self.doc.add_root(fig)

@ray.remote
class BokehServer:
    def __init__(self, market, *args, **kwargs):
        """
        Market Visualizer
        Visualizes a stream of data, automatically refreshes on update()
    
        Args:
            keys (list):       List of object names (str) to be displayed on
                               the market
            history_len (int): How far back to plot data, default is 10 ticks
            title (str):       Title of graph
            x (str):           Name of x value, both in data representation and
                               x axis
            ylabel (str):      Name of y axis on plot
            seed (int):        seed for random number generation
        """
        self.visu   = MarketVisualizer(*args, **kwargs)
        self.market = market

        server = Server(
                {'/': self.init},
                io_loop=IOLoop.current(),
                port=PORT,
                num_procs=1)

        self.thread = None
        self.server = server
        server.start()

        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()

    def init(self, doc):
        self.doc = doc
        self.visu.init(doc)
        self.thread = Thread(target=self.update, args=[])
        self.thread.start()
        self.started = True

    def update(self):
        """
        Update fuction to be used externally.
        data (dict): updated data to be added to market, including tick value
        """
        while True:
            time.sleep(1/60)
            if self.thread is None:
               continue 

            packet = ray.get(self.market.getData.remote())
            data   = deepcopy(self.visu.data)

            for key, val in packet.items():
              data[key].append(val)
            self.visu.data = data
    
            self.doc.add_next_tick_callback(partial(self.stream))

    @gen.coroutine
    def stream(self):
        self.visu.source.stream(self.visu.data, self.visu.history_len)

@ray.remote
class Middleman:
    def __init__(self):
        self.data = {}
        
    def getData(self):
        data = self.data
        self.data = {}
        return data

    def setData(self, data):
        self.data = data


class Market:
    def __init__(self, items, middleman):
        self.items = items
        self.middleman = middleman

        self.data = {}
        self.keys = items

        for i, key in enumerate(self.keys):
            self.data[key] = 0
        self.data['tick'] = 0

        self.tick = 0

    def update(self):
        #Best practice: update in one tick
        for key, val in self.data.items():
            self.data[key] = val
            if key == 'tick':
                self.data[key] = self.tick
            else:
                self.data[key] += 0.2*(random() - 0.5)

        self.tick += 1
        self.middleman.setData.remote(self.data)

# Example setup
PORT=5009
ray.init()
ITEMS = ['Food', 'Water', 'Sword']

middleman  = Middleman.remote()
market     = Market(ITEMS, middleman)
visualizer = BokehServer.remote(middleman, ITEMS)

while True:
  time.sleep(1/30)
  market.update()


