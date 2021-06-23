"""
The MIT License (MIT)

Copyright (c) 2013 Niko Skrypnik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

"""
Loader class
=============

Base loader class which should be used by all other loaders implementations
"""

from kivy.clock import Clock


class BaseLoader(object):

    def __init__(self, **kw):
        self._on_load_start = kw.pop("on_load_start", None)
        self._on_load_progress = kw.pop("on_load_progress", None)
        self._on_load_complete = kw.pop("on_load_complete", None)
        self.source = None

    def on_load_start(self):
        if callable(self._on_load_start):
            self._on_load_start()

    def on_load_progress(self):
        if callable(self._on_load_progress):
            self._on_load_progress()

    def on_load_complete(self):
        if callable(self._on_load_complete):
            self._on_load_complete()

    def __setattr__(self, k, v):
        if k in ["on_load_start", "on_load_progress", "on_load_complete"]:
            if not callable(v):
                raise Exception("%s should be callable" % k)
            setattr(self, "_%s" % k, v)
        else:
            super(BaseLoader, self).__setattr__(k, v)

    def load(self, source, on_load=None, on_progress=None, on_error=None):
        """This function loads objects from source. This function may work
        in both synchronous or asynchronous way. To make it asynchronous
        on_load callback function should be provided.
        """
        self.source = source

        if not callable(on_load):
            return self.parse()

        def _async_load(dt):
            obj = self.parse()
            on_load(obj)

        Clock.schedule_once(_async_load, 0)

    def parse(self):
        """ This should be overridden in subclasses to provide
        parse of the source and return loaded from source object
        """
        raise NotImplementedError('Must be overriden in subclass')
