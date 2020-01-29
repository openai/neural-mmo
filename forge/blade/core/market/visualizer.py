from pdb import set_trace as T

from functools import partial
from threading import Thread
from random import random
import time

from bokeh.models import ColumnDataSource
# from bokeh.models.widgets import CheckboxGroup
from bokeh.plotting import curdoc, figure, show

from tornado import gen


# to run a demo with dummy data:
# $ bokeh serve --show visualizer.py

class MarketVisualizer:
    """
    Market Visualizer
    Visualizes a stream of data, automatically refreshes on update()
    """


    def __init__(self, keys: list, history_len: int = 10,
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
        COLORS = 'blue red green yellow black purple'.split()

        self.history_len = history_len
        self.title       = title
        self.keys        = keys

        self.ylabel      = ylabel
        self.x           = x

        self.data        = {x: [0]}
        self.colors      = {}

        # TODO figure out a better way to ensure each 
        # object has a unique color
        assert len(keys) <= len(COLORS), 'Limited color pool'

        for i, key in enumerate(keys):
            self.data[key] = [0.5]
            self.colors[key] = COLORS[i]

        # this must only be modified from a Bokeh session callback
        self.source = ColumnDataSource(data=self.data)

        # This is important! Save curdoc() to make sure all threads
        # see the same document.
        self.doc = curdoc()
        self.p = figure(
           title='Neural MMO: Market Data',
           x_axis_label=x,
           y_axis_label=ylabel)

        for key in keys:
            self.p.line(x=x, y=key, source=self.source,
                        color=self.colors[key], legend_label=key)

        # Hide lines that you click on in legend,
        # ease of use for visualization
        self.p.legend.click_policy = "hide"
        self.doc.add_root(self.p)
        show(self.p)

        # market plans
        # click on objects in market to display stats about them
        # market overview tab
        # -> trade bandwidth, demand, gdp

    @gen.coroutine
    def stream(self):
        # stream new data to document
        self.source.stream(self.data, self.history_len)

    def update(self):
        """
        Update fuction to be used externally.
        data (dict): updated data to be added to market, including tick value
        """
        # Update the document from callback
        self.doc.add_next_tick_callback(partial(self.stream))

def recv_dummy_data(market_visualizer):
    while True:
        # do some blocking computation
        time.sleep(1)

        #Best practice: update in one tick
        data = {}

        for key, val in market_visualizer.data.items():
            data[key] = val
            # increments tick (x axis)
            if key == mv.x:
                data[key].append(val[-1] + 1)
            # adds r in [-0.5, 0.5) to each value
            else:
                data[key].append(val[-1] + random() - 0.5)
        mv.data = data
        mv.update()

# Example setup
#MARKET_ITEMS = ['Food', 'Water', 'Sword']
#mv = MarketVisualizer(MARKET_ITEMS)

#thread = Thread(target=recv_dummy_data, args=[mv])
#thread.start()
