import numpy as np
import time

t = time.time()
x = np.random.randn(4096, 4096)
np.dot(x, x)
print('Time: ', time.time() - t)

