.. |icon| image:: /resource/icon/icon_pixel.png

.. figure:: /resource/image/splash.png

|icon| Video Demos
##################

`Neural MMO v1.4: A Massively Multiagent Environment for AI Research <https://youtu.be/y_f77u9vlLQ>`_

`Neural MMO v1.3: A Massively Multiagent Game Environment for Training and Evaluating Neural Networks <https://youtu.be/DkHopV1RSxw>`_

`Neural MMO v1.0: A Massively Multiagent Game Environment <https://s3-us-west-2.amazonaws.com/openai-assets/neural-mmo/neural_mmo_client_demo.mp4>`_

`Neural MMO Pre-1.0 <https://youtu.be/tCo8CPHVtUE>`_

|icon| Publications
###################

:download:`[Poster] </resource/update/NMMO_ICML2020_Poster.pdf>` :download:`Neural MMO: Ingredients for Massively Multiagent Artificial Intelligence Research  </resource/update/nmmo_icml2020.pdf>` (ICML 2020 LAOW Workshop) (v1.4)

`Neural MMO v1.3: A Massively Multiagent Game Environment for Training and Evaluating Neural Networks <http://ifaamas.org/Proceedings/aamas2020/pdfs/p2020.pdf>`_ (AAMAS Extended Abstract, 2020)

`Neural MMO v1.3: A Massively Multiagent Game Environment for Training and Evaluating Neural Networks <https://arxiv.org/abs/2001.12004>`_ (arXiv, 2020)

`Neural MMO: A Massively Multiagent Game Environment for Training and Evaluating Intelligent Agents <https://arxiv.org/abs/1903.00784>`_ (arXiv, 2019) (v1.0)

`Neural MMO: A Massively Multiagent Game Environment <https://github.com/jsuarez5341/neural-mmo>`_ (OpenAI Blog, 2019) (v1.0)

|icon| Presentations
####################

`[Slides] <https://docs.google.com/presentation/d/1HYdoe3btw1USWaufBO1yuqFIOg-XW8E2wX0vZal0LtY/edit?usp=sharing>`_ `Neural MMO: A Saga in Deep Reinforcement Learning <https://www.twitch.tv/videos/900545247?t=03h03m06s>`_ (English Week 2021, IUT Vannes) (v1.5)

`Neural MMO v1.3: A Massively Multiagent Game Environment for Training and Evaluating Neural Networks <https://underline.io/lecture/167-neural-mmo-v1.3-a-massively-multiagent-game-environment-for-training-and-evaluating-neural-networks>`_ (AAMAS 2020)

`[Slides] <https://docs.google.com/presentation/d/1GLrvm9ShqDz5whoC0_LUhu0uxnefTQksuE9qc1hXfjg/edit?usp=sharing>`_ `Neural MMO v1.3 Pre-release <https://youtu.be/8iPTrzhB9Yk?t=312>`_ (OxAI VR & AI Virtual Seminar in NeosVR)

|icon| Update Slides
####################

Discontinued in v1.4+ -- better demos and patch notes have made these obsolete.

`Neural MMO v1.3 <https://docs.google.com/presentation/d/1tqm_Do9ph-duqqAlx3r9lI5Nbfb9yUfNEtXk1Qo4zSw/edit?usp=sharing>`_

`Neural MMO v1.2 <https://docs.google.com/presentation/d/1G9fjYS6j8vZMfzCbB90T6ZmdyixTrQJQwZbs8l9HBVo/edit?usp=sharing>`_

`Neural MMO v1.1 <https://docs.google.com/presentation/d/1EXvluWaaReb2_s5L28dOWqyxf6-fvAbtMcBbaMr-Aow/edit?usp=sharing>`_

|icon| Additional Links
#######################

`Ideology <https://docs.google.com/document/d/1_76rYTPtPysSh2_cFFz3Mfso-9VL3_tF5ziaIZ8qmS8/edit?usp=sharing>`_ (Two-pager, whiskey material)

`Style Guide <https://docs.google.com/presentation/d/1m0A65nZCFIQTJm70klQigsX08MRkWcLYea85u83MaZA/edit?usp=sharing>`_ (Website and figure theme)

|icon| Patch Notes + Version History
####################################

The `[OpenAI] <https://github.com/openai/neural-mmo>`_ repository only hosts v1.0. My personal `[Github] <https://github.com/jsuarez5341/neural-mmo>`_ hosts the latest version in *master* and all previous versions as separate branches. This documentation page is generated from the latest environment release. Feel free to drop in the Discord #support channel if you are having trouble. You can expect fast fixes to Github issues and even faster replies to Discord PMs.

.. figure:: /resource/legacy/v1-5_env.png

**v1.5:** Large maps, Dashboard, Scripted Baselines
   - Blade: Full rework to support large environments and scripted players/NPCs
      - Map representation
         - Terrain generation for large maps
         - Environment caching to enable fast resets
         - Tiles are now limited to one occupying agent
         - Reworked tile material enum and properties
      - NPCs
         - Passive: Meanders around the map
         - Neutral: Meanders around the map until attacked, then fights back
         - Hostile: Actively hunts and attacks players and other NPCs
         - Level ranges and spawning locations are configurable for all NPC types
         - Navigation based on A* search
      - Scripted Baselines
         - Extension of the NPC AI module to support scripted player policies
         - Fixed-horizon food/water min-max search with Dijkstra's algorithm and dynamic programming backends
         - Intentional exploration capabilities enable broad coverage of large and small maps
      - Equipment
         - NPCs spawn with chestplates/platelegs of a level appropriate for their skills
         - Players/NPCs wearing equipment drop it upon death
         - Players automatically equip any items better than their current items
         - Equipment provides a large bonus to defense
         - Reworked combat formulas to account for this new system
   - Trinity: New home for non-neural-specific infrastructure and tools
      - Serialized observations
         - Maintains a flat tensor representation of the environment state
         - This representation is kept synchronous with the game state representation
         - Each entity (Player/Tile) is represented as discrete and continuous vectors
         - Observations are computed by slicing from tensor representations without traversing game objects
         - Discrete values are flat-indexed for ease of use in embedding layers
      - Evaluation
         - Runs the given model on multiple maps and aggregates data for the dashboard
         - Outputs a tabular summary of the results for baselines and publications
         - Usable on training maps, held-out evaluation maps (default), and transfer maps
      - Dashboard
         - Environment log function records customizable data for customizable plot types whenever an agent dies
         - Data is aggregated during training and at the end of evaluation
         - Bokeh dashboard is built using the aggregated data for the specified plot types
         - Dashboard is rendered in an interactive browser session
   - Ethyr: Simplified attribute processing
      - The Trinity additions flatten the bottom layer of the observation hierarchy
      - This removes a slow loop and significant complexity from IO embedding/unembed modules
      - We have standardized on the Recurrent baseline architecture for this release
   - Embyr: Full rework to support large environments and scripted players/NPCs
      - Map representation
         - All terrain representation code has been rewritten using the performant Unity Entity Component System
         - Tiles are loaded into and welded together in chunks
         - Lava/water assets have been replaced with more performant variants
      - Visuals
         - Tile textures are now configurable with the hifi (default)/medfi/lofi command
         - Attack animations have been replaced with more distinctive and aesthetic assets
         - A graphical bug causing sharp normals in some tile models has been fixed
         - UI and console retouched to match the new website theme
   - /projekt: Demo code for evaluation, overlays and logging
      - Unified command-line utility for map generation, training, evaluation, visualization, and rendering
      - Experiment config for canonical large/small baseline tasks
      - Single-file ~400 line RLlib wrapper/demo
      - Non-RLlib specific code has been moved to Trinity
      - Improved overall code cohesion and quality

.. figure:: /resource/legacy/v1-4_env.png

**v1.4:** RLlib Support and Overlays
   - Blade: Minor API changes have been made for compatibility with Gym and RLlib
      - Exposed the registerOverlay() and getValStim() methods for writing custom overlays
      - Environment reset method now returns only obs instead of (obs, rewards, dones, infos)
      - Environment obs and dones are now both dictionaries keyed by agent ids rather than agent game objects
      - The IO modules from v1.3 now delegates batching to the user, e.g. RLlib. As such, several potential sources of error have been removed
      - A bug allowing agents to use melee combat from farther away than intended has been fixed
      - Minor range and damage balancing has been performed across all three combat styles
   - Trinity: This module has been temporarily shelved
      - Now hosts the Twisted server code for interfacing with the client
      - Core functionality has been ported to RLlib in collaboration with the developers
      - We are working with the RLlib developers to add additional features essential to the long-term scalability of Neural MMO
      - The Trinity/Ascend namespace will likely be revived in later infrastructure expansions. For now, the stability of RLlib makes delegating infrastructure pragmatic to enable us to focus on environment development, baseline models, and research
   - Ethyr: Proper NN building blocks for complex worlds
      - Streamlined IO, memory, and attention modules for use in building PyTorch policies
      - A high-quality pretrained baseline reproducible at the scale of a single desktop
   - Embyr: Overlay shaders for visualizing learned policies
      - Pressing tab now brings up an in-game console
      - A help menu lists several shader options for visualizing exploration, attention, and learned value functions
      - Shaders are rendered over the environment in real-time with partial transparency
      - It is no longer necessary to start the client and server in a particular order
      - The client no longer needs to be relaunched when the server restarts
      - Agents now turn smoothly towards their direction of movement and targeted adversaries
      - A graphical bug causing some agent attacks to render at ground level has been fixed
      - Moved twistedserver.py into the main neural-mmo repository to better separate client and server
      - Confirmed working on Ubuntu, MacOS, and Windows + WSL
   - /projekt: Demo code fully rewritten for RLlib
      - The new demo is much shorter, approximately 250 lines of code
      - State-of-the-art LSTM + self-attention based policy trained with distributed PPO
      - Batched GPU evaluation for real-time rendering
      - Trains in a few hours on a reasonably good desktop (5 rollout worker cores, 1 underutilized GTX 1080Ti GPU)
      - To avoid introducing RLlib into the base environment as a hard dependency, we provide a small wrapper class over Realm using RLlib's environment types
      - Attempted to migrate from a pip requirements.txt to Poetry for streamlined dependency management, but Poetry is still too buggy at the present.
      - We have migrated configuration to Google Fire for improved command line argument parsing

**v1.3:** Prebuilt IO Libraries
   - Blade: We have improved and streamlined the previously unstable and difficult to use IO libraries and migrated them here. The new API provides framework-agnostic IO.inputs and IO.outputs functions that handle all batching, normalization, serialization. Combined with the prebuilt IO networks in Ethyr, these enable seamless interactions with an otherwise complex structured underlying environment interface. We have made corresponding extensions to the OpenAI Gym API to support variable length actions and arguments, as well as to better signal episode boundaries (e.g. agent deaths). The Quickstart guide has been updated to cover this new functionality as part of the core API.
   - Trinity: Official support for sharding environment observations across multiple remote servers; performance and logging improvements.
   - Ethyr: A Pytorch library for dynamically assembling hierarchical attention networks for processing NMMO IO spaces. We provide a few default attention modules, but users are also free to use their own building blocks -- our library can handle any well defined PyTorch network. We have taken care to separate this PyTorch specific functionality from the core IO libraries in Blade: users should find it straightforward to extend our approach to TensorFlow and other deep learning frameworks.
   - Embyr: Agents now display additional information overhead, such as when they are immune to attacks or when they have been frozen in place.
   - A reasonable 8-population baseline model trained on 12 (old) CPU cores in a day.
   - Improved and expanded official documentation
   - New tutorials covering distributed computation and the IO API
   - The Discord has grown to 80+! Join for active development updates, the quickest support, and community discussions.

.. figure:: /resource/legacy/v1-2_env.png

**v1.2:** Unity Client and Skilling
   - Blade: Skilling/professions. This persistent progression system comprises Hunting, Fishing (gathering skills) and Constitution, Melee, Range, Mage (combat skills). Skills are improved through usage: agents that spend a lot of time gathering resources will become able to gather and store more resources at a time. Agents that spend a lot of time fighting will be able to inflict and take more damage. Additional bug fixes and enhancements.
   - Trinity: Major new infrastructure API: Ascend -- a generalization of Trinity. Whereas v1.1 Trinity implemented cluster, server, and node layer APIs with persistence, synchronous/asynchronous, etc... Ascend implements a single infrastructure "layer" object with all the same features and more. Trinity is still around and functions identically -- it has just been reimplemented in ~10 lines of Ascend. Additional bug fixes and features; notable: moved environment out of Trinity.
   - Ethyr: Streamlined and simplified IO api. Experience manager classes have been redesigned around v1.2 preferred environment placement, which places the environment server side and only communicates serialized observations and actions -- not full rollouts. Expect further changes in the next update -- IO is the single most technically complex aspect of this project and has the largest impact on performance.
   - Embyr: Focus of this update. Full client rewrite in Unity3D with improved visuals, UI, and controls. The new client makes visualizing policies and tracking down bugs substantially easier. As the environment progresses towards a more complete MMO, development entirely in THREE.js was impractical. This update will also speed up environment development by easing integration into the front end.
   - Baseline model is improved but still weak. This is largely a compute issue. I expect the final model to be relatively efficient to train, but I'm currently low on processing power for running parallel experiments. I'll be regaining cluster access soon.
   - Official documentation has been updated accordingly
   - 20+ people have joined the Discord. I've started posting frequent dev updates and thoughts here.

**v1.1:** Infrastructure and API rework, official documentation and Discord
   - Blade: Merge Native and VecEnv environment API. New API is closer to Gym
   - Trinity: featherweight CPU + GPU infrastructure built on top of Ray and engineered for maximum flexibility. The differences between Rapid style training, tiered MPI gradient aggregation, and even the v1.0 CPU infrastructure are all minor usage details under Trinity.
   - Ethyr: New IO api makes it easy to interact with the complex input and output spaces of the environment. Also includes a killer rollout manager with inbuilt batching and serialization for communication across hardware.
   - Official github.io documentation and API reference
   - Official Discord
   - End to end training source. There is also a pretrained model, but it's just a weak single population foraging baseline around 2.5x of random reward. I'm currently between cluster access -- once I get my hands on some better hardware, I'll retune hyperparameters for the new demo model.


.. figure:: /resource/legacy/v1-0_env.png

**v1.0:** Initial OpenAI environment release
   - Blade: Base environment with foraging and combat
   - Embyr: THREE.js web client
   - Trinity: CPU based distributed training infrastructure
   - Ethyr: Contrib library of research utilities
   - Basic project-level documentation
   - End to end training source and a pretrained model

.. figure:: /resource/legacy/v0-2_env.png

**v0.x:** Private development
   - Personal-scale side project and early prototyping

.. figure:: /resource/legacy/v0-1_env.jpg
