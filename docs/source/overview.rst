.. |ags| image:: docs/source/resource/ags.png
.. |env| image:: docs/source/resource/splash.png

.. |air| image:: docs/source/resource/air_thumbnail.png
.. |earth| image:: docs/source/resource/earth_thumbnail.png
.. |fire| image:: docs/source/resource/fire_thumbnail.png
.. |water| image:: docs/source/resource/water_thumbnail.png

|env|

.. #####################################
.. WARNING: Do NOT edit the overview.rst. That file gets copied from the root README.rst and will be overwritten
.. #####################################

|ags| Welcome to the Platform!
##############################

**Important:** My `personal github <https://github.com/jsuarez5341/neural-mmo>`_ hosts the most updated version of the environment and 3D web client. The `OpenAI <https://github.com/openai/neural-mmo>`_ repo will continue to host major environment patches, but it does not get minor bugfixes. 

**Getting Started:** Neural MMO is a Gym environment but unlike most platforms, it's persistent, multi-(and variable numbered)-agent, and has nontrivial input/output spaces. The quickest way to dive in is:

1. Check out the `quickstart guide <https://jsuarez5341.github.io/neural-mmo/build/html/neural-mmo.html#>`_ for setup and the `update slide deck <https://docs.google.com/presentation/d/1EXvluWaaReb2_s5L28dOWqyxf6-fvAbtMcBbaMr-Aow/edit?usp=sharing>`_ to see what's new.

2. Skim the new `documentation <https://jsuarez5341.github.io/neural-mmo/build/html/autodoc/forge.trinity.html>`_ for Trinity/Pantheon/God/Sword. The core API is fully documented. The Ethyr contrib has partial documentation -- any confusing omissions will be prioritized upon request.

3. Head over to the `/projekt <https://github.com/jsuarez5341/neural-mmo/tree/master/projekt>`_ in the source and familiarize yourself with the end-to-end "demo". This is my personal research code on the platform.

4. Join our `Discord <https://discord.gg/BkMmFUC>`_ for help and discussion!

This is an open source project. Contributions are encouraged. I actively review issues and PRs.

|ags| Overview
==============

This environment is the first neural MMO; it attempts to create agents that scale to real world complexity. Simulating evolution on Earth is computationally infeasible, but we can construct a reasonable and efficient facsimile. We consider MMORPGs (Massive Multiplayer Online Role Playing Games) the best proxy for the real world among human games: they are complete macrocosms featuring thousands of agents per persistent world, diverse skilling systems, global economies, and ad-hoc high stakes single and team based conflict.

|ags| Version History
=====================

v1.1: Infrastructure and API rework, official documentation and Discord
   - Blade: Merge Native and VecEnv environment API. New API is closer to Gym
   - Trinity: featherweight CPU + GPU infrastructure built on top of Ray and engineered for maximum flexibility. The differences between Rapid style training, tiered MPI gradient aggregation, and even the v1.0 CPU infrastructure are all minor usage details under Trinity.
   - Ethyr: New IO api makes it easy to interact with the complex input and output spaces of the environment. Also includes a killer rollout manager with inbuilt batching and serialization for communication across hardware.
   - Official github.io documentation and API reference
   - Official Discord
   - End to end training source. There is also a pretrained model, but it's just a weak single population foraging baseline around 2.5x of random reward. I'm currently between cluster access -- once I get my hands on some better hardware, I'll retune hyperparameters for the new demo model.

v1.0: Initial OpenAI environment release
   - Blade: Base environment with foraging and combat
   - Embyr: THREE.js web client
   - Trinity: CPU based distributed training infrastructure
   - Ethyr: Contrib library of research utilities
   - Basic project-level documentation
   - End to end training source and a pretrained model

Note: This is an open source project, but it's a young one. For as long as I'm still running it solo, every minor version should be considered a "soft release" -- bits of documentation will be missing and some APIs will throw internal error messages. However, you can expect any issues raised on Github to be addressed quickly, and I actively monitor the Discord support channel. Feel free to drop in if you are having trouble.

|ags| Namesake
==============

In formal publications, we refer to our project as simply a "Neural MMO." Internally and informally, we call it "Projekt: Godsword." The name comes from two sources: CD Projekt Red, my personal favorite game dev studio, and OldSchool Runescape, which contains an iconic set of weapons called god swords. The latter is a particularly good model for AI environments; the former is more of a soft flavor inspiration.

|ags| Disclaimer
================

I originally began this problem independently. I continued working on it and released v1.0 during a 6 month internship and collaboration with OpenAI. The client was a collaboration between myself and Clare Zhu. The environment has since been my main project. I plan to continue developing it as an EECS PhD candidate at MIT under Phillip Isola until someone convinces me that there is a better way to solve AGI.

Everything written in the source and documentation is my own opinion. I do not speak for OpenAI, MIT, Clare, or Phillip.

|ags| Authorship and License
============================

I, `Joseph Suarez <https://github.com/jsuarez5341>`_, am the author of the environment code base. Yilun Du assisted with running experiments and particularly in setting up tournaments. Phillip Isola and Igor Mordatch have been invaluable collaborators and advisers throughout the project. The environment is registered to OpenAI and available under the MIT license. There is a smaller original code base and game kernel that I (Joseph Suarez) retain ownership of, along with associated ideas. I created these before my employment -- the initial commit of the OpenAI neural-mmo repository represents the latest pre-employment timestep.

The client was developed independently as a collaboration between myself and Clare Zhu. It was originally created as follow-up work for the paper and blog post, but we ended up merging it in. This is also the reason that the project is split into two repositories. It is registered to us jointly and available under the MIT license.

|ags| Assets
============

Some assets used in this project belong to `Jagex <https://www.jagex.com/en-GB/>`_, the creators of Runescape, such as

|ags| |earth| |water| |fire| |air|

We currently use them for flavor as an homage to the game that inspired the project. We believe these fall under fair use as a not-for-profit project for the advancement of artificial intelligence research -- however, we are more than happy to remove them upon request. We do own the 2D and 3D files for agents.

.. image:: docs/source/resource/neuralRED.png
.. image:: docs/source/resource/neuralBLUE.png
.. image:: docs/source/resource/neuralGREEN.png
.. image:: docs/source/resource/neuralFUCHSIA.png
.. image:: docs/source/resource/neuralORANGE.png
.. image:: docs/source/resource/neuralMINT.png
.. image:: docs/source/resource/neuralPURPLE.png
.. image:: docs/source/resource/neuralSPRING.png
.. image:: docs/source/resource/neuralYELLOW.png
.. image:: docs/source/resource/neuralCYAN.png
.. image:: docs/source/resource/neuralMAGENTA.png
.. image:: docs/source/resource/neuralSKY.png

