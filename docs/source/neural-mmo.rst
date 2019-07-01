.. |ags| image:: resource/ags.png
.. |env| image:: resource/banner.png

.. |air| image:: resource/air_thumbnail.png
.. |earth| image:: resource/earth_thumbnail.png
.. |fire| image:: resource/fire_thumbnail.png
.. |water| image:: resource/water_thumbnail.png

|env|

|ags| Quickstart
################

The master branch will always contain the latest stable version. **Users should not fork cowboy branches.** These are hyperagressive dev branches for contributors, not bleeding edge builds.

.. code-block:: python

   #Install OpenAI environment, the Embyr client, and THREE.js. ~1 GB
   git clone https://github.com/jsuarez5341/neural-mmo
   cd neural-mmo

   #We assume you already have a Python 3.7+ Anaconda setup
   python scripts/setup.py

All configuration options are available in experiments.py. The environment is framework independent. So are some of the contrib libraries. Our demo code is a full fledged research pipeline written in PyTorch. It makes use of some PyTorch-specific contrib libraries. If you don't have a strong framework preference, extending the demo is a great way to get started immediately. To run the environment:

.. code-block:: python

   #Run the environment with rendering on
   #This will automatically set --local
   python Forge.py --render

   #Train with 2 GPUs, 2 environments per GPU (editable in config.py)
   #You may need to tweak CUDA_VISIBLE_DEVICES for your machine
   python Forge.py

   #You can also tell Ray to execute locally.
   #This is useful for debugging and works with pdb
   python Forge.py --ray local 


With rendering enabled, navigate to http://localhost:8080/forge/embyr/ in Firefox or Chrome to pull up the Embyr client (`source <https://github.com/jsuarez5341/neural-mmo-client>`_)

|ags| Projekt 
=============

The project is divided into four modules:

======================  ========================
Engineering             Research
======================  ========================
|earth| Blade: Env      |water| Trinity: API
|fire|  Embyr: Render   |air| Ethyr: Contrib
======================  ========================

The objective is to create agents that scale to the complexity and robustness of the real world. This is a variant phrasing of "artificial life." A key perspective of the project is decoupling this statement into subproblems that are concrete, feasible, and directly composable to solve the whole problem. We split the objective into "agents that scale to their environment" and "environments that scale to the real world." These are large respective research and engineering problems, but unlike the original objective, they are specific enough to attempt individually. See Ideology if you find this sort of macro view interesting. 

|water| |air| Research: Agents that scale to env complexity

|earth| |fire| Engineering: Env that scales to real world complexity

|water| Trinity
---------------

The environment itself follows the OpenAI Gym API almost identically.

.. code-block:: python

   from forge.blade.core.realm import Realm
   env = Realm(config, args, mapIdx)

   #The environment is persistent: call reset only upon initialization
   obs = env.reset()

   #Observations contain entity and stimulus
   #for each agent in each environment.
   actions = your_algorithm_here(obs)

   #Observations length is variable (as is the number of agents)
   #The environment is persistent: "dones" denotes whether
   #whether the given agent has died, but the env goes on.
   obs, rewards, dones, infos = env.step(actions)

The Trinity API consists of three base classes --- Pantheon, God, and Sword (see Namesake if that sounds odd) -- that implement framwork agnostic distributed infrastrure on top of the environment. Basic functionality is:

.. code-block:: python

   #Create a Trinity object specifying
   #Cluster, Server, and Core level execution
   trinity = Trinity(Pantheon, God, Sword)
   trinity.init(config, args)
   while theSkyIsBlue:
      trinity.step()

You override Pantheon, God, and Sword to specify functionality at the Cluster, Server, and Core levels, respectively. All communications are handled internally. The demo in /projekt shows how Trinity can be used for Openai Rapid style training with very little code. 

|air| Ethyr
-----------
Ethyr is the "contrib" for this project. It contains useful research tools for interacting with the project. I've seeded it with the helper classes from my personal experiments, including a model save/load manager, a rollout objects, and a basic optimizer. If you would like to contribute code (in any framework, not just PyTorch), please submit a pull request.

|earth| Blade
-------------
Blade is the core environment, including game state and control flow. Researchers should not need to touch this.

|fire| Embyr
------------
`Embyr <https://github.com/jsuarez5341/neural-mmo-client>`_ is an independent repository containing THREE.js web client. It's written in javascript, but it reads like python. This is to allow researchers with a Python background and 30 minutes of javascript experience to begin contributing immediately. You will need to refresh the page whenever you reboot the server (Forge.py). Performance should no longer be an issue, but it runs better on Chrome than Firefox. Other browsers may work but are not officially supported.

I personally plan on continuing development on both the main environment and the client. The environment repo is quite clean, but the client could use some restructuring. I intend to refactor it for v1.2. Environment updates will most likely be released in larger chunks, potentially coupled to future publications. On the other hand, the client is under active and rapid development. You can expect most features, at least in so far as they are applicable to the current environment build, to be released as soon as they are stable. Feel free to contact me with ideas and feature requests.

|ags| Known Limitations
^^^^^^^^^^^^^^^^^^^^^^^

The client has been tested with Firefox on Ubuntu. Don't use Chrome. It should work on other Linux distros and on Macs -- if you run into issues, let me know.

Use Nvidia drivers if your hardware setup allows. The only real requirement is support for more that 16 textures per shader. This is only required for the Counts visualizer -- you'll know your setup is wrong if the terrain map vanishes when switching overlays.

This is because the research overlays are written as raw glsl shaders, which you probably don't want to try to edit. In particular, the counts exploration visualizer hard codes eight textures corresponding to exploration maps. This exceeds the number of allowable textures. I will look into fixing this into future if there is significant demand. If you happen to be a shader wizard with spare time, feel free to submit a PR.
