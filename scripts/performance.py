from forge.blade.core import realm
from forge.blade.core import Config
import time

config = Config()
env = realm.Realm(config, [], 0)

def spawn():
   return None, 0, 'Neural'

env.spawn = spawn

n = 10000
start = time.time()
for i in range(n):
   env.step({})

end = time.time()
print('FPS/core: ', n / (end - start))


