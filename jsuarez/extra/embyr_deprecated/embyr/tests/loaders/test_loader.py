
import os
import unittest
from kivy3.loaders.loader import BaseLoader
from tests.utils import Spy, Sandbox
from kivy.clock import Clock


this_dir = os.path.abspath(os.path.dirname(__file__))


class BaseLoadertestCase(unittest.TestCase):

    def setUp(self):
        self.loader = BaseLoader()
        self.sandbox = Sandbox()

    def tearDown(self):
        self.sandbox.restore()

    def test_load_when_no_on_load(self):
        loader = self.loader
        loader.parse = Spy()
        loader.load('somefile')
        self.assertTrue(loader.parse.is_called())

    def test_on_load_called(self):
        loader = self.loader
        loader.parse = Spy()
        _on_load = Spy()
        # mock Clock.schedule_once
        self.sandbox.stub(Clock, 'schedule_once', call_fake=lambda x, t: x(0))
        loader.load('somesource', on_load=_on_load)
        self.assertTrue(_on_load.is_called(), 'on_load callback should be called')



if __name__ == '__main__':
    unittest.main()
