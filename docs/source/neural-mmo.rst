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

   #Install OpenAI environment and the Embyr client
   git clone https://github.com/jsuarez5341/neural-mmo
   cd neural-mmo

   #We assume you already have a Python 3.7+ Anaconda setup
   python scripts/setup.py

All configuration options are available in experiments.py. The environment is framework independent. So are some of the contrib libraries. Our demo code is a full fledged research pipeline written in PyTorch. It makes use of some PyTorch-specific contrib libraries. If you don't have a strong framework preference, extending the demo is a great way to get started immediately. To run the environment:

.. code-block:: python

   #Train with 4 CPU cores (editable in config.py)
   python Forge.py

   #You can also tell Ray to execute locally.
   #This is useful for debugging and works with pdb
   python Forge.py --ray local 

   #Run the environment with rendering on
   python Forge.py --render


Once the environment is running with --render, run ./client to launch the Unity3D `client <https://github.com/jsuarez5341/neural-mmo-client>`_. For now, you will need to relaunch the client any time you relaunch the server.

|ags| Projekt 
=============

The project is divided into four modules:

=============================  =======================
Engineering                    Research
=============================  =======================
|earth| Blade: Environment     |water| Trinity: API   
|fire|  Embyr: Render          |air| Ethyr: Contrib   
=============================  =======================

The objective is similar to "artificial life": create agents that scale to the complexity and robustness of the real world. A key perspective of the project is decoupling this statement into subproblems that are concrete, feasible, and directly composable to solve the whole problem. We split the objective into "agents that scale to their environment" and "environments that scale to the real world." These are large respective research and engineering problems, but unlike the original objective, they are specific enough to attempt individually. See Ideology if you find this sort of macro view interesting. 

|water| |air| Research: Agents that scale to env complexity

|earth| |fire| Engineering: Env that scales to real world complexity

|water| Trinity
---------------

Neural MMO uses the OpenAI Gym API function signatures:

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

However, there are some necesary deviations in argument/return values:

1. Observations and actions are objects, not tensors. This is a major compute saver, but it also complicates IO -- the process of inputting observations into networks and outputing action choices. Ethyr provides a dedicated IO api to assist with this.

2. The environment supports a large and variable number of agents. Observations are returned with variable length in an arbitrary order. Each observation is tagged with the ID of the associated agent.

3. The environment is ill suited to per-frame rendering and instead functions as an MMO client/server. Example usage is provided in Forge.py.

You can provide your own infrastructure or use our Trinity API. Trinity is a simple three layer persistent, synchronous/asynchronous, distributed computation model that allows you to specify cluster, server, and core level functionality by implementing three base classes. High level usage is:

.. code-block:: python

   #Ready: Create a Trinity object specifying
   #Cluster, Server, and Core level execution
   trinity = Trinity(Pantheon, God, Sword)

   #Aim: Pass it an experiment configuration
   trinity.init(config)

   #Fire.
   while not solved(AGI):
      trinity.step()

Where Pantheon, God, and Sword (see Namesake if that sounds odd) are user defined subclasses of Ascend -- our lightweight and framework agnostic Ray wrapper defining an arbitrary "layer" of infrastructure. All communications are handled internally and easily exposed for debugging. The demo in /projekt shows how Trinity can be used for distributed training with very little code outside of the model and rollout collection. 

|air| Ethyr
-----------
Ethyr is the "contrib" for this project. It contains useful research tools for interacting with the project, most notably IO classes for pre/post processing observations and actions. I've seeded it with the helper classes from my personal experiments, including a model save/load manager, a rollout objects, and a basic optimizer. If you would like to contribute code (in any framework, not just PyTorch), please submit a pull request.

|earth| Blade
-------------
Blade is the core environment, including game state and control flow. Researchers should not need to touch this.

|fire| Embyr
------------
`Embyr <https://github.com/jsuarez5341/neural-mmo-client>`_ is an independent repository containing the Unity3D client. All associated scripts are written in C# but reads relatively similarly to python. Researchers familiar with python and static typing should have no trouble beginning to contribute immediately, even without direct experience in C#. Performance should not be an issue on any decent machine; post in the Discord if you are having issues. 

I am actively developing the environment and associated client in tandem. Updates are typically released in large chunks every few months. The Discrd is the best place to get more frequent news. Feel free to contact me there with ideas and feature requests.

The Legacy THREE.js web client is still available on old branches but does not work with v1.2+ server code. It's written in javascript, but it reads like python. This is to allow researchers with a Python background and 30 minutes of javascript experience to begin contributing immediately. You will need to refresh the page whenever you reboot the server (Forge.py). Performance should no longer be an issue, but it runs better on Chrome than Firefox. Other browsers may work but are not officially supported.
