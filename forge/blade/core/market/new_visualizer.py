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

# to run a demo with dummy data:
# $ bokeh serve --show visualizer.py

@ray.remote
class MarketVisualizer:
    """
    Market Visualizer
    Visualizes a stream of data, automatically refreshes on update()
    """

    def __init__(self, market, keys, history_len: int = 10,
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
        self.packet      = {}
        self.title       = title
        self.keys        = keys

        self.ylabel      = ylabel
        self.x           = x

        self.data        = {x: [0]}
        self.colors      = {}

        self.started = False
        self.market = market
        self.server = self.makeServer()

    def makeServer(self):
       io_loop = IOLoop.current()
       server  = Server({'/': self.init}, io_loop=io_loop, 
               port=PORT, num_procs=1)
       self.server = server
       server.start()

       server.io_loop.add_callback(server.show, "/")
       #server.io_loop.add_callback(self.update, '/')
       server.io_loop.start()

    def init(self, doc):
        # TODO figure out a better way to ensure each 
        # object has a unique color
        assert len(self.keys) <= len(self.COLORS), 'Limited color pool'

        for i, key in enumerate(self.keys):
            self.data[key] = [0.5]
        self.data['tick'] = [0.5]

        # this must only be modified from a Bokeh session callback
        self.source = ColumnDataSource(data=self.data)

        # This is important! Save curdoc() to make sure all threads
        # see the same document.
        #self.doc = curdoc()
        self.doc = doc

        self.p = figure(
           title='Neural MMO: Market Data',
           x_axis_label=self.x,
           y_axis_label=self.ylabel)

        for i, key in enumerate(self.keys):
            self.colors[key] = self.COLORS[i]
            self.p.line(x=self.x, y=key, source=self.source,
                color=self.colors[key], legend_label=key)

        # Hide lines that you click on in legend,
        # ease of use for visualization
        self.p.legend.click_policy = "hide"
        self.doc.add_root(self.p)
        # market plans
        # click on objects in market to display stats about them
        # market overview tab
        # -> trade bandwidth, demand, gdp

        thread = Thread(target=self.update, args=[])
        thread.start()
        print('INITIALIZED')
        self.started = True

    @gen.coroutine
    def stream(self):
        #self.source = ColumnDataSource(data=self.data)
        data = deepcopy(self.data)
        for key, val in self.packet.items():
           data[key].append(val)

        self.data = data

        self.source.stream(self.data, self.history_len)

    def update(self):
        """
        Update fuction to be used externally.
        data (dict): updated data to be added to market, including tick value
        """
        while True:
            time.sleep(0.2)
            print('Update')
            if not self.started:
               continue 

            #Causes sync issues
            packet = ray.get(self.market.getData.remote())
            self.packet = packet

            self.doc.add_next_tick_callback(partial(self.stream))

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
            self.data[key] = [0.5]
        self.data['tick'] = [0.5]

        self.tick = 0

    def update(self):
        data = {}
        #Best practice: update in one tick
        self.tick += 1

        for key, val in self.data.items():
            data[key] = val
            if key == 'tick':
                data[key] = self.tick
            else:
                data[key] = random() - 0.5

        self.data = data
        self.middleman.setData.remote(data)


def update():
    while True:
        time.sleep(1.0)
        print('tick')
        market.update()

# Example setup
PORT=5008
ITEMS = ['Food', 'Water', 'Sword']
ray.init()
middleman  = Middleman.remote()
market     = Market(ITEMS, middleman)
visualizer = MarketVisualizer.remote(middleman, ITEMS)
update()

#thread = Thread(target=update, args=[market])
#thread.start()

#market.update()
#market.update()
#market.update()
#market.update()
#server = Server({'/': thunk})
#server.start()

#server.io_loop.add_callback(server.show, "/")
#server.io_loop.start()
