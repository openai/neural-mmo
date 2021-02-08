.. |icon| image:: /resource/icon/icon_pixel.png

.. |ags| image:: /resource/icon/rs/ags.png
.. |air| image:: /resource/icon/rs/air.png
.. |earth| image:: /resource/icon/rs/earth.png
.. |fire| image:: /resource/icon/rs/fire.png
.. |water| image:: /resource/icon/rs/water.png

.. figure:: /resource/image/splash.png

|icon| Foreword
###############

We're actively looking for academic collaborators and open source contributors for all aspects of the project, including those that do not involve any AI or ML. Message me on Discord if you are interested in getting involved. This guide serves as a brief organizational overview of Neural MMO for new contributors and links to a scrum board. It also contains information on some of the more important technical decisions in this project, some of which may be important to you depending on which portion of the project you are interested in.

|icon| Code Structure
#####################

Neural MMO is broken into four major modules

|air| forge/ethyr: Neural networks. This module contains all code with PyTorch dependencies and associated tools.

|water| forge/trinity: Infrastructure. This module contains everything positions between the environment code and the model code, including the core API, evaluation and visualization tools, and serialized observation representations.

|earth| forge/blade: Core game. This module contains the environment code with minimal dependencies on non-game elements.

|fire| forge/embyr: 3D Unity client. This module contains the C# Unity project and associated executables.

Forge.py is the main file and the RLlib demo is implemented in /projekt

|icon| Tech Stack
#################

**High-Level:** The server is written in Python 3.7. The client is a Unity3D project written in C#. The client is not required for training and is only used for rendering visualizations. These two layers communicate with each other through a Twisted WebSocket server. The documentation is written in Sphinx.

**Server:** Broken into four modules as described in the User Guide. The biggest chunk of code is responsible for the environment game logic, as well as general purpose observation and action processing required for the OpenAI Gym derivative User API. The baselines are PyTorch models and our demo training code uses RLlib. The environment itself does not depend on PyTorch or RLlib (and shouldn't), but they are more or less mandatory if you want an out-of-the-box experience. There isn't a way around this with current frameworks. No frameworks currently support automatic input/output network definitions from complex Gym spaces. The best we could do is to replicate the current design in other frameworks (e.g. TensorFlow). That is too much work for me to do as an individual for too little benefit, but if you want to implement a TensorFlow port, go ahead and I'll merge to forge/ethyr/tensorflow.

We use a slightly customized GoogleFire wrapper for environment and experiment configuration files (pure python with nice CLI support) and a pip requirements.txt for dependency management and installation

**Client:** Embyr is the Neural MMO renderer. The v1.0 client was written in THREE.js. Since then, we have migrated to Unity3D with a C# code base. The renderer functions much like an MMO game client: rather than directly simulating game logic, it renders the current game state from packets communicated by the Neural MMO server over a Twisted WebSocket. This design cuts out the overhead of running a bulky game engine during training and also enables us to keep the environment in pure Python for faster development. Embyr is maintained in a separate repository for historical reasons as well as because it is large and not required on remote servers during distributed training. The terrain is made of a large number of snappable 3D tiles modeled in Blender. The only significant library we use outside of base Unity is websocket-sharp for communicating with the server.

**Dashboard:** The interactive Dashboard is implemented as a Bokeh server and renders in-browser.

**Documentation:** We use Sphinx to autogenerate documentation. The User API is created manually in docs/source/public. The Developer API is a dump of all doctrings using Sphinx Autodoc. The website template is defined in source/_templates and is a reskinned readthedocs.io theme.

|icon| Style
############

We mostly follow PEP8. The only notable exceptions are three-space indents and camel case names -- you don't need to follow these in new modules, but avoid mixing styles or autoformatting large swaths of old code.

The Neural MMO: `[Style Guide] <https://docs.google.com/presentation/d/1m0A65nZCFIQTJm70klQigsX08MRkWcLYea85u83MaZA/edit?usp=sharing>`_ provides canonical color schemes for the website, figures, and presentations. It is not designed for traditional light-on-dark publications -- arXiv and conference papers should significantly downplay fonts/colors to avoid overly high-contrast figures.

|icon| Scrum Board
##################

We manage major improvements and expansions through Github Projects as of v1.5. This is a non-exhaustive list. Feel free to propose new efforts. In general, we're always looking for:
  - New game content in the style of traditional MMOs
  - Methods for hooking up a wider variety of game content to our IO code
  - Better evaluation and visualization tools
  - Better baseline models and more efficient trainers
  - Expansions that support new research directions
  - Client performance improvements
  - Client UI and asset improvements
  - New types of in-client visualizations
  - Improvements to our documentation

|icon| Client Map Representation
################################

When the client first connects to the server, it receives a packet containing the material types of all tiles. In order to build the map, the client selects the correct snappable 3D asset for each tile depending on its material and surrounding tiles. This loading process occurs in chunks of configurable size, and each chunk is welded together into a single object after being loaded. Maps in v1.5+ can have over a million tiles -- performance is crucial. We leverage Unity's DOTS/ECS (Data Oriented Tech Stac/Entity Component System), which allows us to represent tiles/chunks with structs instead of heavier GameObjects. However, this system is not perfect. We cannot change tile types without reloading an entire chunk. As such, the food resource (forest tile) is represented using a separate 3d model. These are also represented using ECS, but it still places a significant load on the client. Large maps can also contain up to 2048 entities (players and NPCs). This has started to become a performance bottleneck as well, and we have not yet integrated ECS into the player logic. ECS is great for performance, but it makes development significantly more complicated, as we essentially lose the conveniences of OOP.

|icon| Observation Representation
#################################

Agent observations are represented by a set of sets: they observe a set of nearby objects (agents and tiles) each parameterized by on the order of a dozen attributes (continuous and discrete values). This quickly becomes a lot of data: for an agent with a vision range of 7, they observe 15x15=256 tiles and up to 256 agents for a total of 512 entities. Multiply this value by up to a thousand agents per environment and we have a lot of data. More importantly, we have a lot of nested object traversals in order to extract the data for each observation from the environment. This is ridiculously slow -- before the following optimizations, observation processing consumed 98+ percent of environment computation time. The solution was to keep a serialized flat-tensor representation of the environment synchronized with the actual environment. Every time the environment updates one of the properties that is observable by agents, the change is reflected in an underlying tensor representation. This allows us to extract agent observations as flat tensor slices. This logic is in forge/trinity/dataframe.py. Be warned: it is essential to follow the patterns used by the Tile and Entity classes to avoid desync. The worst training bugs in Neural MMO invariably come from a mismatch between the game object state and the serialized state.

|icon| Model IO
###############

Each agent observes discrete and continuous tensors for each objects type (currently Tiles and Entities). Discrete values have been flat-indexed to fit a single embedding layer. This enables us to compute discrete/continuous embedding vectors using a single lookup/weight multiply per entity type. The embeddings are then passed to an attentional preprocessor which squashes the variable-length set of objects to a fixed-size representation. It may then be passed to a standard model, currently an LSTM, before being fed to the action model. In order to support variable-length actions such as targeting nearby agents, we use a hard-attentional mechanism. That is, the model hidden state is keyed (dot producted) with action argument embeddings. This allows us to keep the entire model end-to-end differentiable.
