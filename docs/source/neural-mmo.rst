.. |ags| image:: resource/ags.png
.. |env| image:: resource/banner.png

.. |air| image:: resource/air_thumbnail.png
.. |earth| image:: resource/earth_thumbnail.png
.. |fire| image:: resource/fire_thumbnail.png
.. |water| image:: resource/water_thumbnail.png

|env|

|ags| Quickstart
################

**Installation:** The master branch will always contain the latest stable version. *Users should not fork cowboy branches.* These are hyperagressive dev branches for contributors. They are not bleeding edge builds and may be flammable.

.. code-block:: python

   #Download the Neural MMO environment
   #We assume a Python 3.7+ setup with Anaconda pip
   git clone https://github.com/jsuarez5341/neural-mmo && cd neural-mmo

   #Installation options:
   #   --no-client:   Omit the Embyr 3D Unity client
   #   --virtual-env: Install in a virtual env
   python scripts/setup.py

   #Run the pretrained demo model to test the installation
   python Forge.py --render

   #Open the client in a separate terminal
   #You will need to rerun this if you restart the environment
   ./client.sh

|ags| Training from scratch
###########################

Next, we will get familiar with the baseline parameters and train a model from scratch. Open up experiments.py, which contains all of the training configuration options. First, we'll disable the test mode flags:

.. code-block:: python

   LOAD = False
   TEST = False
   BEST = False

Our baseline was trained on a 12 core machine. Your CPU probably does not have this many. To use 4 cores instead:

.. code-block:: python

   NGOD = 4

Now we can train a model:

.. code-block:: python

   python Forge.py

If you leave it running, you will see the reward steadily increasing. The baseline model gets to >23 average lifetime after training for several days on 12 cores. Once you are satisfied, enable testing flags and run with rendering enabled to view learned policies. Learning not to run into lava is a good sanity check.

|ags| The IO API
################

On the surface, Neural MMO follows the OpenAI Gym API:

.. code-block:: python

from forge.blade.core.realm import Realm
from experiments import Experiment, Config

if __name__ == '__main__':
   config = Experiment('demo', Config).init()

   env = Realm(config, *args)
   obs, rewards, dones, infos = env.reset()

   while not done:
      actions = somePolicy(obs)
      obs, rewards, dones, info = env.step(actions)

However, the actual contents of *obs, rewards, dones, info* is nonstandard by necessity. Gym isn't built for multiagent environments -- and certainly not for ones with complex hierarchical observation and action spaces. You're free to develop your own methods for handling these, but we've already done all that work for you. Let's make use of the core IO libraries:

.. code-block:: python

from forge.blade.core.realm import Realm
from experiments import Experiment, Config

if __name__ == '__main__':
   config = Experiment('demo', Config).init()

   env = Realm(config, *args)
   obs, rewards, dones, infos = env.reset()

   while not done:
      input, _ = io.inputs(obs, rewards, dones, *args)
      output   = somePolicy(input)

      actions = io.outputs(output)
      obs, rewards, dones, info = env.step(actions)

We're almost done. The IO API handles batching, normalization, and serialization. The only remaining issue is that *somePolicy* must handle hierarchical data and variable action spaces. Let's use the Ethyr prebuilt IO modules:

.. code-block:: python

from forge.blade.core.realm import Realm
from experiments import Experiment, Config
import torch

if __name__ == '__main__':
   config = Experiment('demo', Config).init()

   env = Realm(config, *args)
   obs, rewards, dones, infos = env.reset()

   policy = torch.nn.Sequential(
      ethyr.Input(*args),
      ethyr.Output(*args)

   while not done:
      input, _ = io.inputs(obs, rewards, dones, *args)
      output   = policy(input)

      actions = io.outputs(output)
      obs, rewards, dones, info = env.step(actions)

And there you have it! You can insert your own model between the input and output networks without having to deal with nonstandard structured data. However, this only covers the forward pass. We haven't discussed rollout collection, training, or any population based methods. For a fully featured and well documented example, hop over to /projekt in the environment repo.

|ags| Distributed computation with Ascend
#########################################

Ascend is a lightweight wrapper on top of the excellent Ray distributed computing library. The core paradigm is to model each *layer* of hardware -- cluster, server, core -- by subclassing the Ascend object. Let's first implement a remote client (Sword) without using Ascend. In order to keep track of several remote clients, we will also create a server (God).

.. code-block:: python

import ray, time

@ray.remote
class Sword:
   def __init__(self, idx):
      self.idx = idx

   def step(self):
      time.sleep(1)
      return self.idx

class God:
   def __init__(self, n=5):
      self.disciples = [Sword.remote(i) for i in range(n)]

   def step(self):
      clientData = ray.get([d.step.remote() for d in self.disciples])
      print(clientData) #[0, 1, 2, 3, 4]

if __name__ == '__main__':
   ray.init()
   God().step()

Ascend enables us to do all of this without manually writing loops over hardware:

.. code-block:: python

from forge.trinity.ascend import Ascend
import ray, time

@ray.remote
class Sword:
   def __init__(self, idx):
      self.idx = idx

   def step(self):
      time.sleep(1)
      return self.idx

class God(Ascend):
   def __init__(self, n=5):
      super().__init__(Sword, n)

   def step(self):
      clientData = super().step()
      print(clientData) #[0, 1, 2, 3, 4]

if __name__ == '__main__':
   ray.init()
   God().step()

The source is only a few hundred lines and isn't very useful in toy examples. Ascend really shines in more complex environments that already have too many moving parts:

.. code-block:: python

from forge.trinity.ascend import Ascend, runtime, waittime
import ray, time

@ray.remote
class Sword(Ascend):
   def __init__(self, idx):
      super().__init__(None, 0)
      self.idx = idx

   @runtime
   def step(self, coef, bias):
      time.sleep(1)
      return coef*self.idx + bias

class God(Ascend):
   def __init__(self, n=5):
      super().__init__(Sword, n)

   def update(self):
      time.sleep(1)

   @runtime
   def step(self):
      asyncHandles = super().distrib(
            2,
            [4, 3, 2, 1, 0],
            shard=(False, True))

      self.update()
      clientData = super().sync(asyncHandles)
      print(clientData) #[4, 5, 6, 7, 8]

if __name__ == '__main__':
   ray.init()
   God().step()

Like before, we have a server interacting with five remote clients. This time, the *coef* argument is shared among clients while the *bias* argument is sharded among them. Additionally, we are using the computation time of the clients to perform additional work in the server side *update()* function. And we are also logging performance statistics, specifically time spent performing useful computation vs time spent waiting, for both layers. The Neural MMO demo has a third infrastructure layer for the cluster. Even in this toy example, Ascend is saving us quite a bit of code. In a full research environment, we have found it an indispensable tool. Welcome, Ascendant!
