.. |env| image:: /resource/image/v1-4_splash.png
.. |icon| image:: /resource/icon/icon_pixel.png

.. |ags| image:: /resource/icon/rs/ags.png
.. |air| image:: /resource/icon/rs/air.png
.. |earth| image:: /resource/icon/rs/earth.png
.. |fire| image:: /resource/icon/rs/fire.png
.. |water| image:: /resource/icon/rs/water.png

.. role:: python(code)
    :language: python

|env|

|icon| Abstract
###############

`[Demo Video] <https://youtu.be/y_f77u9vlLQ>`_ | `[Github] <https://github.com/jsuarez5341/neural-mmo>`_ | `[Discord] <https://discord.gg/BkMmFUC>`_ | `[Twitter] <https://twitter.com/jsuarez5341>`_

Progress in multiagent intelligence research is fundamentally limited by the complexity of environments available for study. Neural MMO is a massively multiagent AI research environment inspired by Massively Multiplayer Online (MMO) role playing games -- self-contained worlds featuring thousands of agents per persistent macrocosm, diverse skilling systems, local and global economies, complex emergent social structures, and ad-hoc high-stakes single and team based conflict.  Our goal is not to simulate the near-infinite physical processes of life on Earth but instead to construct an efficient facsimile that incentivizes the emergence of high-level social and general artificial intelligence. To this end, we consider MMOs the best proxy for the real world among human games.

|icon| Installation
###################

**Dependencies:** Anaconda Python 3.7.x and gcc. Tested for Ubuntu 16.04/18.04/20.04, macOS Catalina (10.15), and Windows 10.

Ubuntu and macOS:

.. code-block:: python

   #Download Neural MMO and run the pretrained demo model
   git clone https://github.com/jsuarez5341/neural-mmo && cd neural-mmo
   bash scripts/setup.sh
   python Forge.py

   #Open the client in a separate terminal
   ./client.sh

Windows + WSL:

.. code-block:: python

   #Execute on WSL Ubuntu (dependencies required)
   git clone https://github.com/jsuarez5341/neural-mmo && cd neural-mmo
   bash scripts/setup.sh --SERVER_ONLY
   python Forge.py

   #Execute on Windows (dependencies not required)
   git clone https://github.com/jsuarez5341/neural-mmo-client
   neural-mmo-client/UnityClient/neural-mmo.exe

**Troubleshooting:**
  - Forge.py takes ~30 seconds to launch the server. This is due to overlay precomputation; you can speed it up by running with ``--COMPUTE_GLOBAL_VALUES=False``. This will also fix most out-of-memory errors.
  - If PyTorch is not recognizing your GPU, you can run CPU only using ``CUDA_VISIBLE_DEVICES="" python Forge.py``, but expect low FPS.
  - Most compatibility issues with the client and unsupported operating systems can be resolved by opening the project in the Unity Editor.
  - If none of the above work, post in #support on Discord

**Versioning:** The master branch will always contain the latest stable version. Each previous version release is archived in a separate branch. Other branches are for contributors and developers only: they are not bleeding edge builds and may be flammable.

|icon| Overview
###############

Agents in Neural MMO progress persistent skills while exploring procedurally generated terrain and engaging in strategic combat over thousands of timesteps. Our platform is both an environment and a set of tools for visualizing and comparing emergent policies of intelligent agents.

.. image:: /resource/figure/web/overview.svg

The long-term goal of Neural MMO is to train artificial general intelligence in simulation -- that is, **agents that scale to the complexity of the real world**. The project is divided into research and engineering modules that cleanly segment this objective into concrete and approachable research and engineering tasks:

Research
--------

**Agents that scale to the complexity of their environment**

|water| Trinity: Distributed computation framework based on Ray+RLlib

|air| Ethyr: Baseline models and research utility contrib -- submit PRs with your own tools!

Engineering
-----------

**Environments that scale to the complexity of the real world**

|earth| Blade: Core game environment and extended OpenAI Gym external API

|fire| Embyr: 3D Unity game client for test-time visualization

|icon| Getting Started
######################

Neural MMO extends the OpenAI Gym API to support additional environment complexity: persistence, large/variable agent populations, and hierarchical observation/action spaces. The quickest way to dive in is:

**1:** Work through the tutorials below and familiarize yourself with the `[Realm API] <https://jsuarez5341.github.io/neural-mmo/build/html/autodoc/forge.blade.core.realm.html>`_

**2:** Join our `[Discord] <https://discord.gg/BkMmFUC>`_ community for help and discussion. **This is the best way to contact me**

**3:** Develop your own fork and contribute your features to the platform.

Neural MMO is fully open-source -- to succeed long-term, we will need the help of talented researchers, software engineers, game designers, and technical artists. I actively review issues and pull requests.

|icon| Training from scratch
############################

Next, we will get familiar with the baseline parameters and train a model from scratch. Open up projekt/config.py, which contains all of the training configuration options. You can either edit defaults here or override individual parameters using command line arguments. To train a baseline, simply run:

.. code-block:: python

  python Forge.py --RENDER=False --MODEL=None

You can reduce batch size if you are running out of memory or disable CUDA if you don't have a GPU on hand, but performance may suffer. All baseline models train overnight with four i7-9700K CPU cores @3.6 GHz + one GTX 1080Ti at very low utilization and 32 GB of RAM:

.. image:: /resource/figure/web/train.png

As a sanity check, your agents should have learned not to run into lava after several epochs, around 20 average lifetime. The trained baseline models range within 30-40 average lifetime fully trained. However, individual agents may live much longer -- we have seen >10,000 ticks (~100 minutes real-time). Additionally, higher average lifetime is not always strictly better -- the performance of each agent is loosely coupled to the performance of all other agents. Rendering and overlays help resolve discrepancies.

|icon| Rendering and Overlays
#############################

Embyr is the Neural MMO renderer. It is written in C# using Unity3D and functions much like an MMO game client: rather than directly simulating game logic, it renders the current game state from packets communicated by the Neural MMO server over a Twisted WebSocket. This design cuts out the overhead of running a bulky game engine during training and also enables us to keep the environment in pure Python for faster development. Embyr is maintained in a separate repository for historical reasons as well as because it is large and not required on remote servers during distributed training. Agents advance various foraging and combat skills by collecting food and water and engaging in fights with other agents:

.. image:: /resource/image/v1-4_combat.png

To view an agent's skill levels or follow it with the camera, simply click on it:

.. image:: /resource/image/v1-4_ui.png

The client ships with an in-game console (press tilde ~ to toggle) stocked with prebuilt overlays for visualizing various aspects of the learned policy.

.. image:: /resource/figure/web/overlays.svg


The counts overlay renders a heatmap of agent exploration in real time:

.. image:: /resource/image/v1-4_counts.png

The attention overlay renders egocentric heatmaps of each agent's attention weightings in real time:

.. image:: /resource/image/v1-4_attention.png

The values overlay renders a heatmap of the agent's learned value function in real time:

.. image:: /resource/image/v1-4_values.png

The globalValues overlay hallucinates an agent on each cell and computes the value function for that agent with no other agents on the map and all resources present. This requires a forward pass for each of the ~3600 tiles in the environment. The overlay is precomputed once during server initialization (~30 seconds) and may be disabled in projekt/config.py for faster startup:

.. image:: /resource/image/v1-4_globalValues.png

You can also write your own overlays using Realm.registerOverlay(). For example, the value function overlay is implemented as:

.. code-block:: python

   def values(self, obs):
      '''Computes a local value function by painting tiles as agents
      walk over them. This is fast and does not require additional
      network forward passes'''
      for idx, agentID in enumerate(obs):
         r, c = self.realm.desciples[agentID].base.pos
         self.valueMap[r, c] = float(self.model.value_function()[idx])

      colorized = overlay.twoTone(self.valueMap)
      self.realm.registerOverlay(colorized, 'values')

Custom overlays can make full use of the current environment state, but note that this is not part of the official API. See /projekt/overlays.py for full implementations of the baseline overlays.

|icon| The IO API
#################

OpenAI Gym supports standard definitions for structured, mixed discrete/continuous observation and action (input/output or IO) spaces. However, there are a few issues:

1. OpenAI Gym has a couple of blind spots surrounding dictionary and repeated set observations

2. The existence of structured IO spaces does not imply a corresponding neural architecture for processing them

Neural MMO resolves both of these problems out of the box. We have worked with the RLlib developers to augment OpenAI Gym's *spaces* API with two new structure objects, *Repeated* and *FlexDict*.

Additionally, we have implemented substantially general procedural generation code that automatically fits attentional PyTorch architectures to the given IO spaces. These will be subject to minor tweaks from update to update but should remain structurally stable from update to update. The high-level concept is to model observations of sets of entities, each of which is a set of attributes:

.. image:: /resource/figure/web/header.svg

Entity embeddings are created by attending over attributes, and the observation is flattened to a fixed-length embedding by attenting over entity embeddings. Actions are similarly defined by targeting entity embeddings with attention. The diagram below summarizes this process -- see the `[Neural MMO v1.3 white paper] <https://arxiv.org/abs/2001.12004>`_ for details

.. image:: /resource/figure/web/io.svg

Our Baseline models include an abstract *Base* model that instantiates our IO modules but defers the hidden network to subclasses:

.. code-block:: python

   class Base(nn.Module):
      def __init__(self, config):
         ...
         self.output = io.Output(config)
         self.input  = io.Input(config,
               embeddings=policy.BiasedInput,
               attributes=policy.Attention)
         self.valueF = nn.Linear(config.HIDDEN, 1)

      def hidden(self, obs, state=None, lens=None):
         raise NotImplementedError('Implement this method in a subclass')

      def forward(self, obs, state=None, lens=None):
         entityLookup  = self.input(obs)
         hidden, state = self.hidden(entityLookup, state, lens)
         self.value    = self.valueF(hidden).squeeze(1)
         actions       = self.output(hidden, entityLookup)
         return actions, state

Custom models work by defining new subnetworks and overriding the *hidden* method. For example:

.. code-block:: python

   class Simple(Base):
      def __init__(self, config):
         '''Simple baseline model with flat subnetworks'''
         super().__init__(config)
         h = config.HIDDEN

         self.conv   = nn.Conv2d(h, h, 3)
         self.pool   = nn.MaxPool2d(2)
         self.fc     = nn.Linear(h*3*3, h)

         self.proj   = nn.Linear(2*h, h)
         self.attend = policy.Attention(self.embed, h)

      def hidden(self, obs, state=None, lens=None):
         #Attentional agent embedding
         agents, _ = self.attend(obs[Stimulus.Entity])

         #Convolutional tile embedding
         tiles     = obs[Stimulus.Tile]
         self.attn = torch.norm(tiles, p=2, dim=-1)

         w      = self.config.WINDOW
         batch  = tiles.size(0)
         hidden = tiles.size(2)
         tiles  = tiles.reshape(batch, w, w, hidden).permute(0, 3, 1, 2)
         tiles  = self.conv(tiles)
         tiles  = self.pool(tiles)
         tiles  = tiles.reshape(batch, -1)
         tiles  = self.fc(tiles)

         hidden = torch.cat((agents, tiles), dim=-1)
         hidden = self.proj(hidden)
         return hidden, state

You can write your own PyTorch models using the same template. Or, if you prefer, you can use our IO subnetworks directly, as is done in our *Base* class. Neural MMO's IO spaces themselves are framework agnostic, but if you want to train in e.g. TensorFlow, you will have to write analogous IO networks.
