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

**Getting Started:** Neural MMO follows the OpenAI Gym API, but unlike most platforms, it's persistent, multi-(and variable numbered)-agent, and has nontrivial input/output spaces. The quickest way to dive in is:

1. Check out the `quickstart guide <https://jsuarez5341.github.io/neural-mmo/build/html/neural-mmo.html#>`_ for setup and the patch notes + slides below to see what's new.

2. Skim the new `documentation <https://jsuarez5341.github.io/neural-mmo/build/html/autodoc/forge.trinity.html>`_ for Trinity/Pantheon/God/Sword. The core API is fully documented. The Ethyr contrib has partial documentation -- any confusing omissions will be prioritized upon request.

3. Head over to the `/projekt <https://github.com/jsuarez5341/neural-mmo/tree/master/projekt>`_ in the source and familiarize yourself with the end-to-end demo. This is my personal research code on the platform.

4. Join our `Discord <https://discord.gg/BkMmFUC>`_ for help and discussion! **This is also the best way to contact me.**

Neural MMO is an open source project. Contributions are encouraged. I actively review issues and PRs.

|ags| Overview
==============

This environment is the first neural MMO; it attempts to create agents that scale to real world complexity. Simulating evolution on Earth is computationally infeasible, but we can construct a reasonable and efficient facsimile. We consider MMORPGs (Massive Multiplayer Online Role Playing Games) the best proxy for the real world among human games: they are complete macrocosms featuring thousands of agents per persistent world, diverse skilling systems, global economies, complex emergent social structures, and ad-hoc high stakes single and team based conflict.

|ags| Version History
=====================

v1.2: Client and Skilling update | `Update Slide Deck <https://docs.google.com/presentation/d/1G9fjYS6j8vZMfzCbB90T6ZmdyixTrQJQwZbs8l9HBVo/edit?usp=sharing>`_
   - Blade: Skilling/professions. This persistent progression system comprises Hunting, Fishing (gathering skills) and Constitution, Melee, Range, Mage (combat skills). Skills are improved through usage: agents that spend a lot of time gathering resources will become able to gather and store more resources at a time. Agents that spend a lot of time fighting will be able to inflict and take more damage. Additional bug fixes and enhancements.
   - Trinity: Major new infrastructure API: Ascend -- a generalization of Trinity. Whereas v1.1 Trinity implemented cluster, server, and node layer APIs with persistence, synchronous/asynchronous, etc... Ascend implements a single infrastructure "layer" object with all the same features and more. Trinity is still around and functions identically -- it has just been reimplemented in ~10 lines of Ascend. Additional bug fixes and features; notable: moved environment out of Trinity.
   - Ethyr: Streamlined and simplified IO api. Experience manager classes have been redesigned around v1.2 preferred environment placement, which places the environment server side and only communicates serialized observations and actions -- not full rollouts. Expect further changes in the next update -- IO is the single most technically complex aspect of this project and has the largest impact on performance.
   - Embyr: Focus of this update. Full client rewrite in Unity3D with improved visuals, UI, and controls. The new client makes visualizing policies and tracking down bugs substantially easier. As the environment progresses towards a more complete MMO, development entirely in THREE.js was impractical. This update will also speed up environment development by easing integration into the front end.
   - Baseline model is improved but still weak. This is largely a compute issue. I expect the final model to be relatively efficient to train, but I'm currently low on processing power for running parallel experiments. I'll be regaining cluster access soon.
   - Official documentation has been updated accordingly
   - 20+ people have joined the Discord. I've started posting frequent dev updates and thoughts here.

v1.1: Infrastructure and API rework, official documentation and Discord | `Update Slide Deck <https://docs.google.com/presentation/d/1EXvluWaaReb2_s5L28dOWqyxf6-fvAbtMcBbaMr-Aow/edit?usp=sharing>`_ 
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

Note: This is an open source project, but it's a young one. For as long as I'm still running it solo, every minor version should be considered a "soft release" -- bits of documentation will be missing and some APIs will throw internal error messages. However, you can expect fast fixes to Github issues and even faster replies to Discord PMs. Feel free to drop in the Discord #support channel if you are having trouble.

|ags| Namesake
==============

In formal publications, we refer to our project as simply a "Neural MMO." Internally and informally, we call it "Projekt: Godsword." The name comes from two sources: CD Projekt Red, my personal favorite game dev studio, and OldSchool Runescape, which contains an iconic set of weapons called god swords. The latter is a particularly good model for AI environments; the former is more of a soft flavor inspiration.

|ags| Authorship, License, Disclaimer
=====================================

I, `Joseph Suarez <https://github.com/jsuarez5341>`_, began this project independently and am the author of the environment code base, which I continued developing at OpenAI. There, Yilun Du assisted with running experiments and particularly in setting up tournaments for the v1.0 release. Phillip Isola and Igor Mordatch have been invaluable collaborators and advisers through v1.0. I continued working on the environment independently thereafter. The environment has since been my main project; I released v1.1 and v1.2 of both the environment and client independently. I plan to continue developing it as an EECS PhD candidate at MIT under Phillip Isola until someone convinces me that there is a better way to solve AGI.

The v1.0 environment is registered to OpenAI and available under the MIT license. v1.1 and v1.2 are derivative works. There is a smaller original code base and game kernel that I (Joseph Suarez) retain ownership of, along with associated ideas. I created these before my employment -- the initial commit of the OpenAI neural-mmo repository represents the latest pre-employment timestep.

The legacy THREE.js client was developed independently as a collaboration between myself and Clare Zhu. It was originally created as follow-up work for the paper and blog post, but we ended up merging it in. This is also the reason that the project is split into two repositories. It is registered to us jointly and available under the MIT license.

Everything written in the source and documentation is my own opinion. I do not speak for OpenAI, MIT, Clare, Phillip, or anyone else involved in the project.

|ags| Assets
============

Some assets used in this project belong to `Jagex <https://www.jagex.com/en-GB/>`_, the creators of Runescape, such as

|ags| |earth| |water| |fire| |air|

We currently use them for flavor as an homage to the game that inspired the project. We believe these fall under fair use as a not-for-profit project for the advancement of artificial intelligence research -- however, we are more than happy to remove them upon request. We do own the 2D and 3D files for agents, represented by three neurons.

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

