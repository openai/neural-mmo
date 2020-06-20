from pdb import set_trace as T
from matplotlib import pyplot as plt
import numpy as np
import re

dat = open('train_attn.txt').read()
R   = []

regex = re.compile('Lifetime: .*')
for line in dat.splitlines():
   lifetime = regex.findall(line)
   if len(lifetime) > 0:
      lifetime = float(lifetime[0][10:])
      R.append(lifetime)

final = []
for i in range(0, len(R), 6):
   final.append(np.mean(R[i:i+6]))
   
#np.save('data_recur.npy', final)
plt.plot(final, 'b', lw=2)
plt.xlabel('Epochs')
plt.ylabel('Lifetime')
plt.title('Randomized Envs')
plt.show()

   
