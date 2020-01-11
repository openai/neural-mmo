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

**Overview:** Neural MMO is a massively multiagent AI research environment inspired by Massively Multiplayer Online Role Playing Games (MMORPGS or MMOs). The long-term goal of our platform is to enable artificial agents to scale to real world intelligence. Simulating the physical processes of evolution on Earth is computationally infeasible, but we can construct a reasonable and effecient facsimile. MMOs are complete macrocosms featuring thousands of agents per persistent world, diverse skilling systems, local and global economies, complex emergent social structures, and ad-hoc high stakes single and team based conflict. We consider them the best proxy for the real world among human games.


**Versioning:** The `OpenAI <https://github.com/openai/neural-mmo>`_ only hosts v1.0. My personal fork `personal github <https://github.com/jsuarez5341/neural-mmo>`_ hosts the latest version in *master* and all previous versions as separate branches. This documentation page is generated from the latest environment release.

**Getting Started:** Neural MMO extends the OpenAI Gym API to support additional environment complexity: persistence, large/variable agent populations, and hierarchical observation/action spaces. The quickest way to dive in is:

**1:** Check out the `Quickstart Guide <https://jsuarez5341.github.io/neural-mmo/build/html/neural-mmo.html#>`_ 

**2:** Skim the `Documentation <https://jsuarez5341.github.io/neural-mmo/build/html/autodoc/modules.html>`_

**3:** Get familiar with the end-to-end demo in `/projekt <https://github.com/jsuarez5341/neural-mmo/tree/master/projekt>`_

**4:** Join our `Discord community <https://discord.gg/BkMmFUC>`_ for help and discussion! **This is also the best way to contact me**

**5:** Contribute to the platform! Neural MMO is an active open source project -- I actively review issues and PRs

|ags| Version History
=====================

v1.3: Prebuilt IO Libraries | `[Update Slide Deck] <https://docs.google.com/presentation/d/1tqm_Do9ph-duqqAlx3r9lI5Nbfb9yUfNEtXk1Qo4zSw/edit?usp=sharing>`_
   - Blade: We have improved and streamlined the previously unstable and difficult to use IO libraries and migrated them here. The new API provides framework-agnostic IO.preprocess and IO.postprocess functions that handle all batching, normalization, serialization. Combined with the prebuilt IO networks in Ethyr, these enable seamless interactions with an otherwise complex structured underlying environment interface. We have made corresponding extensions to the OpenAI Gym API to support variable length actions and arguments, as well as to better signal episode boundaries (e.g. agent deaths). The Quickstart guide has been updated to cover this new functionality as part of the core API.
   - Trinity: Official support for sharding environment observations across multiple remote servers; performance and logging improvements.
   - Ethyr: A Pytorch library for dynamically assembling hierarchical attention networks for processing NMMO IO spaces. We provide a few default attention modules, but users are also free to use their own building blocks -- our library can handle any well defined PyTorch network. We have taken care to separate this PyTorch specific functionality from the core IO libraries in Blade: users should find it straightforward to extend our approach to TensorFlow and other deep learning frameworks.
   - Embyr: Agents now display additional information overhead, such as when they are immune to attacks or when they have been frozen in place.
   - A reasonable 8-population baseline model trained on 12 (old) CPU cores in a day.
   - Improved and expanded official documentation and quickstart guide
   - The Discord has grown to 80+! Join for active development updates, the quickest support, and community discussions.

v1.2: Unity Client and Skilling | `[Update Slide Deck] <https://docs.google.com/presentation/d/1G9fjYS6j8vZMfzCbB90T6ZmdyixTrQJQwZbs8l9HBVo/edit?usp=sharing>`_
   - Blade: Skilling/professions. This persistent progression system comprises Hunting, Fishing (gathering skills) and Constitution, Melee, Range, Mage (combat skills). Skills are improved through usage: agents that spend a lot of time gathering resources will become able to gather and store more resources at a time. Agents that spend a lot of time fighting will be able to inflict and take more damage. Additional bug fixes and enhancements.
   - Trinity: Major new infrastructure API: Ascend -- a generalization of Trinity. Whereas v1.1 Trinity implemented cluster, server, and node layer APIs with persistence, synchronous/asynchronous, etc... Ascend implements a single infrastructure "layer" object with all the same features and more. Trinity is still around and functions identically -- it has just been reimplemented in ~10 lines of Ascend. Additional bug fixes and features; notable: moved environment out of Trinity.
   - Ethyr: Streamlined and simplified IO api. Experience manager classes have been redesigned around v1.2 preferred environment placement, which places the environment server side and only communicates serialized observations and actions -- not full rollouts. Expect further changes in the next update -- IO is the single most technically complex aspect of this project and has the largest impact on performance.
   - Embyr: Focus of this update. Full client rewrite in Unity3D with improved visuals, UI, and controls. The new client makes visualizing policies and tracking down bugs substantially easier. As the environment progresses towards a more complete MMO, development entirely in THREE.js was impractical. This update will also speed up environment development by easing integration into the front end.
   - Baseline model is improved but still weak. This is largely a compute issue. I expect the final model to be relatively efficient to train, but I'm currently low on processing power for running parallel experiments. I'll be regaining cluster access soon.
   - Official documentation has been updated accordingly
   - 20+ people have joined the Discord. I've started posting frequent dev updates and thoughts here.

v1.1: Infrastructure and API rework, official documentation and Discord | `[Update Slide Deck] <https://docs.google.com/presentation/d/1EXvluWaaReb2_s5L28dOWqyxf6-fvAbtMcBbaMr-Aow/edit?usp=sharing>`_ 
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

