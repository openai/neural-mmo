import unittest
from utils import Sandbox
from kivy3 import Material


class MaterialTest(unittest.TestCase):

    def setUp(self):
        self.sandbox = Sandbox()
        self.mat = Material()

    def tearDown(self):
        self.sandbox.restore()

    def test_setattr(self):
        self.mat.color = (0., 0., 0.)
        self.assertEquals(self.mat.changes['Ka'], (0., 0., 0.))
        self.mat.shininess = 5
        self.assertEquals(self.mat.changes['Ns'], 5.)

if __name__ == '__main__':
    unittest.main()
