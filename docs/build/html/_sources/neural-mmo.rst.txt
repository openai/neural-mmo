.. |ags| image:: resource/ags.png
.. |env| image:: resource/banner.png

.. |air| image:: resource/air_thumbnail.png
.. |earth| image:: resource/earth_thumbnail.png
.. |fire| image:: resource/fire_thumbnail.png
.. |water| image:: resource/water_thumbnail.png

|env|

|ags| Quickstart
################

The master branch will always contain the latest stable version. Note that master cannot be renamed to the current version because of a limitation in Github Pages. **Don't clone cowboy-dev.** It's an extremely aggressive dev branch for contributors, not a bleeding edge build.

.. code-block:: python

   #Install OpenAI environment, the Embyr client, and THREE.js. ~1 GB
   #Will only install the environment without --recurse-submodules
   git clone https://github.com/jsuarez5341/neural-mmo --recurse-submodules
   cd neural-mmo
   python scripts/setup.py

**We assume you already have an Anaconda setup with Python 3.7+.**. The environment is framework independent, but our experiment code (the demo) uses PyTorch.

To run the environment:

.. code-block:: python

   #Run the environment with rendering on
   python Forge.py --render

   #Train with 2 GPUs, 2 environments per GPU (editable in config.py)
   #You may need to tweak CUDA_VISIBLE_DEVICES per your machine
   bash Forge.sh

   #You can also tell Ray to execute locally.
   #This is useful for debugging and works with pdb
   python Forge.py --ray local 


With rendering enabled, navigate to http://localhost:8080/forge/embyr/ in Firefox or Chrome to pull up the Embyr client (`source <https://github.com/jsuarez5341/neural-mmo-client>`_)

|ags| Projekt 
=============

The project is divided into four modules:

======================  =====================
Engineering             Research
======================  =====================
|earth| Blade: Env      |water| Trinity: API
|fire|  Embyr: Render   |air| Ethyr: Neural
======================  =====================

The objective is to create agents that scale to the complexity and robustness of the real world. This is a variant phrasing of "artificial life." A key perspective of the project is decoupling this statement into subproblems that are concrete, feasible, and directly composable to solve the whole problem. We split the objective into "agents that scale to their environment" and "environments that scale to the real world." These are large respective research and engineering problems, but unlike the original objective, they are specific enough to attempt individually. For a more thorough overview of the project approach and objective, see this `Two Pager <https://docs.google.com/document/d/1_76rYTPtPysSh2_cFFz3Mfso-9VL3_tF5ziaIZ8qmS8/edit?usp=sharing>`_.

|water| |air| Research: Agents that scale to env complexity

|earth| |fire| Engineering: Env that scales to real world complexity

|water| Trinity
---------------

The environment itself follows the OpenAI Gym API almost identically.

.. code-block:: python

   from forge.blade.core.realm import Realm
   env = Realm(config, args, self.step)

   #The environment is persistent: call reset only upon initialization
   obs = env.reset()

   #Observations contain entity and stimulus
   #for each agent in each environment.
   actions = your_algorithm_here(obs)

   #The environment is persistent: "dones" is always None
   #If an observation is missing, that agent has died
   obs, rewards, dones, infos = env.step(actions)

The Trinity API consists of three base classes --- Pantheon, God, and Sword (see Namesake if that sounds odd) -- that implement framwork agnostic distributed infrastrure on top of the environment. Basic functionality is:

.. code-block:: python

   #Create a Trinity object specifying
   #Cluster, Server, and Core level execution
   trinity = Trinity(Pantheon, God, Sword)
   trinity.init(config, args)

You override Pantheon, God, and Sword to specify functionality at the Cluster, Server, and Core levels, respectively. All communications are handled internally. The demo in /projekt shows how Trinity can be used for Openai Rapid style training with very little code. 

|air| Ethyr
-----------
Ethyr is the "contrib" for this project. It contains useful research tools for interacting with the project. I've seeded it with the helper classes from my personal experiments, including a model save/load manager, a rollout objects, and a basic optimizer. If you would like to contribute code (in any framework, not just PyTorch), please submit a pull request.

|earth| Blade
-------------
Blade is the core environment, including game state and control flow. Researchers should not need to touch this, outside perhaps importing core configurations, i/o tools, and enums.

|fire| Embyr
------------
`Embyr <https://github.com/jsuarez5341/neural-mmo-client>`_ is an independent repository containing THREE.js web client. It's written in javascript, but it reads like python. This is to allow researchers with a Python background and 30 minutes of javascript experience to begin contributing immediately. You will need to refresh the page whenever you reboot the server (Forge.py).

Performance is around 50-60 FPS with ~3s load on a high-end desktop, 30 FPS with ~10s load on my Razer laptop. It runs better on Chrome than Firefox. Other browsers may work but are not officially supported.

I personally plan on continuing development on both the main environment and the client. The environment repo is quite clean, but the client could use some restructuring. I intend to refactor it for v1.2. Environment updates will most likely be released in larger chunks, potentially coupled to future publications. On the other hand, the client is under active and rapid development. You can expect most features, at least in so far as they are applicable to the current environment build, to be released as soon as they are stable. Feel free to contact me with ideas and feature requests.

|ags| Known Limitations
^^^^^^^^^^^^^^^^^^^^^^^

The client has been tested with Firefox on Ubuntu. Don't use Chrome. It should work on other Linux distros and on Macs -- if you run into issues, let me know.

Use Nvidia drivers if your hardware setup allows. The only real requirement is support for more that 16 textures per shader. This is only required for the Counts visualizer -- you'll know your setup is wrong if the terrain map vanishes when switching overlays.

This is because the research overlays are written as raw glsl shaders, which you probably don't want to try to edit. In particular, the counts exploration visualizer hard codes eight textures corresponding to exploration maps. This exceeds the number of allowable textures. I will look into fixing this into future if there is significant demand. If you happen to be a shader wizard with spare time, feel free to submit a PR.

|ags| Failure Modes
===================

Evaluation can be somewhat difficult in our setting but is not a major blocker. For smaller experiments, we find population size and resource utilization to be reasonable metrics of success. For larger experiments with sufficient domain randomization, Tournaments (as described in the accompanying paper) allow for cross validation of approaches.

We are currently aware of three failure cases for the project:
  * Computational infeasibility
  * "Agents that scale to their environment" is too hard
  * "Environments that scale to the real world" is too hard

The first failure case is a serious risk, but is shared among all areas of the field. This project is not uniquely compute intensive -- in fact, it is one of few environments where it is straightforward to train reasonable policies on a single CPU. If scale is the main issue here, it is likely shared among most if not all other approaches.

The second problem is probably most familiar to researchers as exploration. Given a cold start, how can agents bootstrap both to better policies and to better exploration strategies? This is a hard problem, but it is unlikely to kill the project because:
  * This is independently an important problem that many researchers are already working on already
  * The environment of this project is designed collaboratively to assist agents early on in learning, rather than adversarially as a hard benchmark
  * `Recent <https://blog.openai.com/openai-five/>`_ `projects <https://blog.openai.com/learning-dexterity/>`_ have demonstrated success at scale.

The third problem probably appears most likely to many researchers, but least likely to anyone who has spent a significant amount of time in MMOs. Here is a map of the NYC subway:

.. image:: resource/quests.png
  :alt: Quest Map
`Source <https://www.reddit.com/user/Gamez_X>`_

Actually, it's a quest map of Runescape, a particular MMO that our environment is loosely based upon. Each quest is a puzzle in itself, takes anywhere from several minutes to several hours to complete, is part of an interconnected web of prerequisites of other quests, and provides different incentives for completion ranging from equipment to unlockable content to experience in a tightly connected set of skills:

.. image:: resource/skills.png
  :alt: Skills

.. image:: resource/equipment.png
  :alt: Equipment
`Source <https://www.jagex.com/en-GB/>`_

In a massive open world:

.. image:: resource/map.png
  :alt: GameMap
`Source <https://www.jagex.com/en-GB/>`_

The most complex class of games considered to date is MOBAs (Massive Online Battle Arenas, e.g. Dota, Quake CTF), which are round based, take on order of an hour, and are mechanically intensive. Achieving 99 in all skills and acquiring the best gear in Runescape takes, at minimum, several thousand hours. In a tournament setting where attacking other players is allowed everywhere, moment-to-moment gameplay is less important than balancing the risks and rewards of any potential strategy--especially in the presence of hundreds of other players attempting to do the same. There is almost certainly still a complexity gap from MMOs to the real world, but we believe it is much smaller than that in environments currently available.

While our environment is nowhere near the level of complexity of a real MMO yet, it does contain key properties of persistence, population scale, and open-endedness. As agents begin to reach the ceiling of the current environment, we plan on continuing development to raise the ceiling.

