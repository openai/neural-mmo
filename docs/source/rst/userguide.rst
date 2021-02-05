.. |icon| image:: /resource/icon/icon_pixel.png

.. role:: python(code)
    :language: python

.. figure:: /resource/image/splash.png

|icon| Introduction
###################

`[Demo Video] <https://youtu.be/y_f77u9vlLQ>`_ | `[User API] <https://jsuarez5341.github.io/neural-mmo/build/html/rst/api.html>`_ | `[Github] <https://github.com/jsuarez5341/neural-mmo>`_ | `[Discord] <https://discord.gg/BkMmFUC>`_ | `[Twitter] <https://twitter.com/jsuarez5341>`_

Progress in multiagent intelligence research is fundamentally limited by the complexity of environments available for study. Neural MMO is a massively multiagent AI research environment inspired by Massively Multiplayer Online (MMO) role playing games -- self-contained worlds featuring thousands of agents per persistent macrocosm, diverse skilling systems, local and global economies, complex emergent social structures, and ad-hoc high-stakes single and team based conflict.  Our goal is not to simulate the near-infinite physical processes of life on Earth but instead to construct an efficient facsimile that incentivizes the emergence of high-level social and general artificial intelligence. To this end, we consider MMOs the best proxy for the real world among human games.

.. figure:: /resource/image/environment.png

Neural MMO ships with pretrained models, scripted baselines, evaluation tools, a customizable dashboard, and an interactive 3D client packed with visualization tools. The guides below contain everything you need to get started. We also run a community `[Discord] <https://discord.gg/BkMmFUC>`_ for support, discussion, and dev updates. This is the best place to contact me.

|icon| Installation
###################

**Versioning:** The master branch will always contain the latest stable version. Each previous version release is archived in a separate branch. Dev branches are not nightly builds and may be flammable.

.. code-block:: python
   :caption: Ubuntu 20.04/18.04 + Anaconda Python 3.8.x

   #Download Neural MMO and run the pretrained demo model
   git clone --depth=1 https://github.com/jsuarez5341/neural-mmo && cd neural-mmo
   bash scripts/setup.sh
   python Forge.py render

   #Open the client in a separate terminal
   ./client.sh

.. code-block:: python
   :caption: Windows 10 with WSl Ubuntu 20.04/18.04 + Anaconda Python 3.8.x

   #Execute on WSL Ubuntu + Anaconda
   git clone --depth=1 https://github.com/jsuarez5341/neural-mmo && cd neural-mmo
   bash scripts/setup.sh --SERVER_ONLY
   python Forge.py render

   #Execute on Windows
   git clone --depth=1 https://github.com/jsuarez5341/neural-mmo-client
   neural-mmo-client/UnityClient/neural-mmo.exe

**Troubleshooting:**
  - Post installation errors in #support on the `[Discord] <https://discord.gg/BkMmFUC>`_
  - Most compatibility issues with the client and unsupported operating systems can be resolved by opening the project in the Unity Editor
  - If you want full commit history, clone without ``--depth=1`` (including in scripts/setup.sh for the client). This flag is only included to cut down on download time

CLI
###

Forge.py is the main file for the included demo and starter project (/projekt). It includes commands for map generation, training, evaluation, visualization, and rendering. To view documentation:

.. code-block:: python

  python Forge.py --help

.. code-block:: text

  NAME
      Forge.py --help - Neural MMO CLI powered by Google Fire

  SYNOPSIS
      Forge.py --help - GROUP | COMMAND

  DESCRIPTION
      Main file for the RLlib demo included with Neural MMO.

      Usage:
         python Forge.py <COMMAND> --config=<CONFIG> --ARG1=<ARG1> ...

      The User API documents core env flags. Additional config options specific
      to this demo are available in projekt/config.py.

      The --config flag may be used to load an entire group of options at once.
      The Debug, SmallMaps, and LargeMaps options are included in this demo with
      the latter being the default -- or write your own in projekt/config.py

  GROUPS
      GROUP is one of the following:

       config
         Large scale Neural MMO training setting

  COMMANDS
      COMMAND is one of the following:

       evaluate
         Evaluate a model on --EVAL_MAPS maps from the training set

       generalize
         Evaluate a model on --EVAL_MAPS maps not seen during training

       generate
         Generate game maps for the current --config setting

       render
         Start a WebSocket server that autoconnects to the 3D Unity client

       train
         Train a model starting with the current value of --MODEL

       visualize
         Web dashboard for the latest evaluation/generalization results


Terrain Generation
##################

We're going to need some maps to play with in the tutorials below. If you're following along interactively and want to keep things quick, we suggest only generating the small maps. Generating image previews of each map can be useful in certain circumstances. The files for large maps are huge, so we'll only generate PNGs for small maps.

.. code-block:: python

  python Forge.py generate --config=SmallMaps --TERRAIN_RENDER
  python Forge.py generate --config=LargeMaps

.. code-block:: text

  Generating 256 training and 64 evaluation maps:
  100%|████████████████████████████████████████████████| 320/320 [01:35<00:00,  3.34it/s]
  Generating 256 training and 64 evaluation maps:
  100%|████████████████████████████████████████████████| 320/320 [09:53<00:00,  1.85s/it]

Generating small maps without rendering takes 5-10 seconds on a modern CPU.

.. figure:: /resource/image/map.png

   Example map from resource/maps/procedural-small/map1/map.png

Terrain generation is controlled by a number of parameters prefixed with TERRAIN_. The config documentation details them all, and you can experiment with larger modifications to the procedural generation source in forge/blade/core/terrain.py.

Rendering and Overlays
######################

Rendering the environment requires launching both a server and a client. To launch the server:

.. code-block:: python

  python Forge.py render --config=SmallMaps

Launch client.sh in a separate shell or click the associated executable. The server will take a few seconds to a minute to load the pretrained policy and connect to the client.

.. figure:: /resource/image/ui.png

   You should see this view once the map loads

The on-screen instructions demonstrate how to pan and zoom in the environment. You can also click on agents to examine their skill levels. The in-game console (which you can toggle with the tilde key) give you access to a number of overlays.

.. image:: /resource/image/overlays.png

The counts (exploration) overlay is computed by splatting the agent's current position to a counts map. Most other overlays are computed analogously. However, you can also do more impressive things with a bit more compute. For example, the tileValues and entityValues overlays simulate an agent on every tile and computes the value function with respect to local tiles/entities. Note that some overlays, such as counts and skills, are well-defined for all models. Others, such as value function and attention, do not exist for scripted baselines.

Writing your own overlays is simple. You can find the source code for general overlays (those computable by scripted baselines) in forge/trinity/overlay.py. RLlib-specific overlays that require access to the trainer/model are included in projekt/rllib_wrapper.py.

Training
########

Evaluating on small/large maps will load the associated pretrained baseline by default. To reproduce our baselines by training from scratch:

.. code-block:: python

  python Forge.py train --config=SmallMaps --MODEL=None
  python Forge.py train --config=LargeMaps --MODEL=None

.. code-block:: text

        ___           ___           ___           ___
       /__/\         /__/\         /__/\         /  /\
       \  \:\       |  |::\       |  |::\       /  /::\     An open source
        \  \:\      |  |:|:\      |  |:|:\     /  /:/\:\    project originally
    _____\__\:\   __|__|:|\:\   __|__|:|\:\   /  /:/  \:\   founded by Joseph Suarez
   /__/::::::::\ /__/::::| \:\ /__/::::| \:\ /__/:/ \__\:\  and formalized at OpenAI
   \  \:\~~\~~\/ \  \:\~~\__\/ \  \:\~~\__\/ \  \:\ /  /:/
    \  \:\  ~~~   \  \:\        \  \:\        \  \:\  /:/   Now developed and
     \  \:\        \  \:\        \  \:\        \  \:\/:/    maintained at MIT in
      \  \:\        \  \:\        \  \:\        \  \::/     Phillip Isola's lab
       \__\/         \__\/         \__\/         \__\/

   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
   ▏Epoch: 16▕▏Sample: 8923.8/s (64.0s)▕▏Train: 35.4/s (235.2s)▕
   ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
      ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
      ▏Population  ▕▏Min:      1.0▕▏Max:    103.0▕▏Mean:     51.6▕▏Std:     21.9▕
      ▏Lifetime    ▕▏Min:      0.0▕▏Max:    998.0▕▏Mean:     50.8▕▏Std:     69.9▕
      ▏Skilling    ▕▏Min:     10.0▕▏Max:     46.5▕▏Mean:     14.3▕▏Std:      4.9▕
      ▏Combat      ▕▏Min:      3.0▕▏Max:     10.0▕▏Mean:      3.2▕▏Std:      0.5▕
      ▏Equipment   ▕▏Min:      0.0▕▏Max:      8.0▕▏Mean:      0.0▕▏Std:      0.1▕
      ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
   ▏Epoch: 17▕▏Sample: 8910.2/s (62.2s)▕▏Train: 33.7/s (227.8s)▕
   ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
      ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
      ▏Population  ▕▏Min:      1.0▕▏Max:    103.0▕▏Mean:     51.6▕▏Std:     21.9▕
      ▏Lifetime    ▕▏Min:      0.0▕▏Max:    998.0▕▏Mean:     50.8▕▏Std:     69.9▕
      ▏Skilling    ▕▏Min:     10.0▕▏Max:     46.5▕▏Mean:     14.3▕▏Std:      4.9▕
      ▏Combat      ▕▏Min:      3.0▕▏Max:     10.0▕▏Mean:      3.2▕▏Std:      0.5▕
      ▏Equipment   ▕▏Min:      0.0▕▏Max:      8.0▕▏Mean:      0.0▕▏Std:      0.1▕
      ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
   ▏Epoch: 18▕▏Sample: 8885.9/s (59.5s)▕▏Train: 32.4/s (217.2s)▕
   ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
      ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
      ▏Population  ▕▏Min:      1.0▕▏Max:    103.0▕▏Mean:     51.6▕▏Std:     21.9▕
      ▏Lifetime    ▕▏Min:      0.0▕▏Max:    998.0▕▏Mean:     50.8▕▏Std:     69.9▕
      ▏Skilling    ▕▏Min:     10.0▕▏Max:     46.5▕▏Mean:     14.3▕▏Std:      4.9▕
      ▏Combat      ▕▏Min:      3.0▕▏Max:     10.0▕▏Mean:      3.2▕▏Std:      0.5▕
      ▏Equipment   ▕▏Min:      0.0▕▏Max:      8.0▕▏Mean:      0.0▕▏Std:      0.1▕
      ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
   ▏Neural MMO v1.5▕▏Epochs: 18.0▕▏kSamples: 236.8▕▏Sample Time: 1022.2▕▏Learn Time: 3797.6▕
   ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

The training monitor above summarizes wall-clock time spent on sampling vs training and displays performance for the last three epochs. You can train reasonably good small-map models in a few hours and decent large-map models overnight on a single desktop with one GPU. See Baselines for exact training times and performances of our models. Specify the --MODEL=current flag throughout the remainder of these tutorials to load the model you just trained.

Note:
  - Any subsequent training commands will overwrite your checkpoint files. We suggest copying them to somewhere safe. Their default location is written in experiment/path.txt.
  - You can modify this path file to point the current model to other checkpoints.
  - The training monitor receives performance updates when environments reset, which is independent of epoch boundaries. As such, multiple contiguous epochs may have identical summary statistics.

Evaluation
##########

Evaluation in open-ended massively multiagent settings is akin to that in the real world. There isn't an obvious single real-number metric. It's like trying to order people from best to worst. Nonetheless, we can still make meaningful insights about agent behavior and draw well-evidenced conclusions about relative performance. This section will introduce you to Neural MMO's suite of evaluation and visualization tools.

To evaluate a pretrained model and a scripted baseline:

.. code-block:: python

  python Forge.py evaluate --config=SmallMaps --EVAL_MAPS=1
  python Forge.py evaluate --config=SmallMaps --EVAL_MAPS=1 --MODEL=scripted

.. code-block:: text

  Number of evaluation maps: 1
  100%|██████████████████████████████████████████████| 1000/1000 [00:32<00:00, 31.10it/s]
  Number of evaluation maps: 1
  100%|██████████████████████████████████████████████| 1000/1000 [01:01<00:00, 16.17it/s]

Note that we have used a single evaluation map here to keep runtime short -- our baselines average over several maps, and you should follow the protocol detailed in Baselines in formal comparisons.

Advanced
********

Neural MMO provides three sets of evaluation settings:

**Training Maps:** Evaluate on the same maps used for training. This is standard practice in reinforcement learning. *Enable by setting the EVAL_GENERALIZE flag to false*

**Evaluation Maps:** Evaluate on a set of held-out maps drawn from the training map *distribution* generated using different random seeds. *This is the default setting*

**Generalization Maps:** Evaluate large-map models on small maps (hard) or small-map models on large maps (very hard). *Enable by setting the appropriate --config*

Dashboard and Statistics
########################

The "evaluate" command stores data from the most recent run in experiment/evaluation.npy. To view a summary:

.. code-block:: python

  python Forge.py visualize #After evaluating pretrained baseline
  python Forge.py visualize #After evaluating scripted baseline

.. code-block:: text

  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
  ▏Population  ▕▏Min:     24.0▕▏Max:     63.0▕▏Mean:     52.6▕▏Std:      4.9▕
  ▏Lifetime    ▕▏Min:      0.0▕▏Max:    981.0▕▏Mean:     52.5▕▏Std:    100.8▕
  ▏Skilling    ▕▏Min:     10.0▕▏Max:     48.0▕▏Mean:     14.9▕▏Std:      5.8▕
  ▏Combat      ▕▏Min:      3.0▕▏Max:     24.0▕▏Mean:      4.6▕▏Std:      2.4▕
  ▏Equipment   ▕▏Min:      0.0▕▏Max:     12.0▕▏Mean:      0.1▕▏Std:      0.9▕
  ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
  ▏Population  ▕▏Min:     25.0▕▏Max:     61.0▕▏Mean:     50.4▕▏Std:      4.1▕
  ▏Lifetime    ▕▏Min:      0.0▕▏Max:    933.0▕▏Mean:     51.8▕▏Std:     67.4▕
  ▏Skilling    ▕▏Min:     10.0▕▏Max:     49.5▕▏Mean:     14.9▕▏Std:      5.5▕
  ▏Combat      ▕▏Min:      3.0▕▏Max:     30.0▕▏Mean:      4.3▕▏Std:      2.7▕
  ▏Equipment   ▕▏Min:      0.0▕▏Max:     18.0▕▏Mean:      0.1▕▏Std:      0.9▕
  ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

From the summary stats, the models look pretty comparable. Since the scripted baseline performs an exact min-max search using a ton of hand-coded domain knowledge, this is actually quite a good result. But it would be nice to have finer-grained insights -- both to aid in future development and for the paper. The "visualize" command also loads a browser-based interactive dashboard:

.. figure:: /resource/image/baselines/SmallMaps/neural_small_maps.png

   Pretrained neural baseline

.. figure:: /resource/image/baselines/SmallMaps/scripted_combat.png

   Scripted baseline

Each row of the dashboard contains multiple visualization styles for one row of the summary table. In this particular instance, the Skill Level bar chart is most illuminating -- notice how the scripted model uses only Ranged combat whereas the pretrained model uses a mix of Ranged and Mage. I set the scripted model to only use range combat because I thought it was probably slightly stronger overall, but apparently Range and Mage are fairly balanced. The pretrained model avoids Melee even though it does the most damage, probably because the current movement system makes it difficult to close distance to an opponent -- perhaps I should consider changing the movement system in a future update.

So, why do we need 15 plots when only one turned out to be important? First of all, we didn't know which plot would highlight an interesting difference ahead of time. Second, there are some smaller observations we can make, such as the pretrained model obtaining significantly more equipment pickups while the scripted model obtained fewer and better pickups (Equipment scatter plots). Or that the pretrained model has a slightly heavier Lifetime right tail, as seen in the Lifetime Gantt plot. Many of our most experiments (and worst bug fixes) were motivated by an unusual disparity in the dashboard.

And before you ask, yes: there's a boring publication theme: specify --VIS_THEME=publication. In fact, you can create custom logging with a highly configurable dashboard to go with it in only a few lines of code -- just override the log method of forge/trinity/env.py to specify your own data tracks and plot styles.

.. figure:: /resource/image/publication_theme.png

   Publication theme
