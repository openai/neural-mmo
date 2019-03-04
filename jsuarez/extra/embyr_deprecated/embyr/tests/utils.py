'''
Module provides some test utilities
'''

class Spy:
    '''
    This is the test helper which helps us to check whether function has been
    called and which parameters have been passed to
    '''
    def __init__(self, call_fake=None, returns=None):
        self.call_count = 0
        self.args = []
        self.call_fake = call_fake
        self.returns = returns

    def __call__(self, *args, **kwargs):
        self.args.append((args, kwargs))
        self.call_count += 1
        if callable(self.call_fake):
            return self.call_fake(*args, **kwargs)
        else:
            if self.returns:
                return self.returns

    def is_called(self):
        return self.call_count > 0


class Sandbox:
    '''
    This little sucker helps us to mock some modules and functions and then
    restore it
    '''

    def __init__(self):
        'Constructor'
        self._storage = {}

    def stub(self, obj, func_name, **kw):
        if not callable(getattr(obj, func_name)):
            raise Exception('You may stub only callable objects')
        # save previous value of stubbed function
        if not obj in self._storage:
            obj_storage = {}
            self._storage[obj] = obj_storage
        else:
            obj_storage = self._storage[obj]

        if func_name in obj_storage:
            raise Exception('%s function has been already stubbed' % func_name)
        # store function into sandbox storage
        obj_storage[func_name] = getattr(obj, func_name)

        setattr(obj, func_name, Spy(**kw))

    def restore(self):
        if not hasattr(self, '_storage'):
            return

        for obj, obj_storage in self._storage.iteritems():
            keys_to_del = []
            for key, original_value in obj_storage.iteritems():
                # restore all original value to an object
                setattr(obj, key, original_value)
                keys_to_del.append(key)
        del self._storage

    def __del__(self):
        'When we destruct object don\'t forget to restore all values'
        self.restore()
