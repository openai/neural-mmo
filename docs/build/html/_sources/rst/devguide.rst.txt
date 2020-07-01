.. |env| image:: /resource/image/v1-4_splash.png
.. |icon| image:: /resource/icon/icon_pixel.png

|env|

|icon| Foreword
###############

The formal User API is new as of v1.4. This means we've just gotten public functionality to the point where it's relatively stable and wont change too much from update to update. The internals are still moving fast and subject to constant refactoring and reworking. This up-front engineering cost is essential to the long-term success of the project and more or less makes it impossible to have a stable developer API for the time being.

That said, we are working on stabilizing and exposing small sections of the API and do expect to have a reasonable developer guide at some point. Neural MMO is fully open source and in no way limited to experienced reinforcement learning researchers. We also need full-stack software engineers and game designers/programmers/technical artists, among others. The purpose of this section in its current form is to provide a starting point for those interested in contributing to the platform. If you think something is missing, let me know on Discord.

|icon| Before You Start
#######################

Neural MMO is open source by traditional measures -- good PRs get merged. In practice, more is required. Neural MMO is neither a pure research nor software engineering endeavor; long-term success requires people from a variety of skill backgrounds to contribute. Here's how you can help:

**Research Science:** We need better baseline models and algorithms. The largest likely improvement will come from rewriting the baseline model to be fully attentional. It should be possible to express the input, hidden, and output networks as Transformer modules. I've done some preliminary work and have a decent Attentional baseline, but it needs some tuning and also doesn't incorporate attention at every layer. Note that the above is intended for contributors who want to improve the base platform. We also support collaborations with researchers who want to use Neural MMO as a platform for more creative explorations into multiagent communication, population-based algorithms, emergent communication, etc. Come chat on Discord.

**Research Engineering:** The single hardest engineering challenge in Neural MMO lies between the model and the environment. Our IO processing allows us to modify the environment observation and action spaces without having to manually rewrite networks. Currently, we only support discrete and continuous observations and discrete fixed and variable length actions. We would like to expand support to include vector-values observations and actions, as well as continous values actions. We also need peripheral research utilities -- better model versioning, versioning, and troubleshooting tools.

**Software Engineering:** The Neural MMO core is nothing more than a traditional game with a few extra design decisions to support efficiency and simplicity as a research environment. Research on Neural MMO is fundamentally limited by the simplicity of the environment -- more complex game mechanics give agents more systems to learn and interact with. We need more content: item and inventory systems, player communication, trade, PvE, interesting terrain generation, etc.

**Game Development:** The server-side code can be handled more or less by traditional software engineering. Game designers and programmers would be helpful there, but experienced Unity developers would probably be more helpful on the client side. The Neural MMO client is currently limited to relatively small maps that are assembled from 3D tiles in what is probably the stupidest way possible. We need scalable systems for handling large maps and large numbers of agents. As the game and rendering logic are separated, we also need to keep up with features being added to the server.

**Technical Artwork:** The models, animations, UIs, and other assets used in the client are good by research environment standard but amateur otherwise. We need help improving all aspects of the client visuals

**Technical Writing:** The website docs could use improvement -- assistance in editing is greatly appreciated.

**Other:** If you have other relevent skills and can improve a portion of this project, we welcome your help. Come chat on Discord about how you can get involved if you don't fit into one of the buckets above or come from a multidisciplinary background.

|icon| Tech Stack
#################

**High-Level:** The server is written in Python 3.7. The client is a Unity3D project written in C#. The client is not required for training and is only used for rendering visualizations. These two layers communicate with each other through a Twisted WebSocket server. The documentation is written in Sphinx.

**Server:** Broken into four modules as described in the User Guide. The biggest chunk of code is responsible for the environment game logic, as well as general purpose observation and action processing required for the OpenAI Gym derivative User API. The baselines are PyTorch models and our demo training code uses RLlib. The environment itself does not depend on PyTorch or RLlib (and shouldn't), but they are more or less mandatory if you want an out-of-the-box experience. There isn't a way around this with current frameworks. No frameworks currently support automatic input/output network definitions from complex Gym spaces. The best we could do is to replicate the current design in other frameworks (e.g. TensorFlow). That is too much work for me to do as an individual for too little benefit, but if you want to implement a TensorFlow port, go ahead and I'll merge to forge/ethyr/tensorflow.

As of v1.4, we use a slightly customized GoogleFire wrapper for environment and experiment configuration files (pure python with nice CLI support) and a pip requirements.txt for dependency management and installation

**Client:** The v1.0 client was written in THREE.js. Since then, we have migrated to Unity3D with a C# code base. The terrain is made of a large number of snappable 3D tiles modeled in Blender. The only significant library we use outside of base Unity is websocket-sharp for communicating with the server.

**Documentation:** We use Sphinx to autogenerate documentation. The User API is created manually in docs/source/public. The Developer API is a dump of all doctrings using Sphinx Autodoc. The website template is defined in source/_templates and is a reskinned readthedocs.io theme.

|icon| Versioning
#################

We assume Python 3.7+ with the packages listed in the requirements.txt, minus Pytorch/RLlib which are only required for the baseline models. Note that this means:

1. Dicts are assumed sorted by insertion order

2. No walrus := operator

|icon| Style
############

Deviations from PEP:

1. 3 space indents

2. Lines cut at 78 characters

Unit tests at the User API level and a basic linter will be required as the project scales. We currently have neither of these. For the time being, verify that your commits do not break the demo policy + renderer. Larger patches should run the training code overnight to ensure agents still learn similar behaviors.

The Neural MMO: `[Style Guide] <https://docs.google.com/presentation/d/1m0A65nZCFIQTJm70klQigsX08MRkWcLYea85u83MaZA/edit?usp=sharing>`_ provides canonical color schemes for the website, figures, and presentations. It is not designed for traditional light-on-dark publications -- arXiv and conference papers should significantly downplay fonts/colors to avoid overly high-contrast figures.
